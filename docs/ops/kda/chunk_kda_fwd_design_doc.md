# KDA Forward Delivery Design Document

This document describes the delivered forward design of Kimi Delta Attention (KDA). It progresses from the problem background through the end-to-end flow, primary call chain, mathematical formulation, current physical implementation, Tensor Processing Unit (TPU) optimizations, supported functionality, and design trade-offs. The mathematical definitions follow Section 3.1 and Appendix B of *Kimi Linear: An Expressive, Efficient Attention Architecture* (arXiv:2510.26692v2). Implementation details reflect the current code behavior.

## 1. Background and Problem Definition

### 1.1 Objective

KDA forward accepts query, key, value, a per-channel decay gate, and a delta-rule write coefficient. It produces an output for each token while maintaining a fixed-size associative memory state. For every batch element and attention head, the state has shape `K x V` and does not grow with sequence length.

The Tokamax TPU backend enters this implementation through:

```python
(o, final_state), residuals = backend._fwd(q, k, v, g, beta, ...)
```

The primary inputs and outputs are:

| Symbol | Tokamax backend shape | Meaning |
|---|---:|---|
| `B` | - | Batch size |
| `T` | - | Input token length; rank-local length under Context Parallelism |
| `H` | - | Number of attention heads |
| `N` | - | Number of logical segments represented in the recurrent-state axis; one for fixed-length input |
| `K` | - | Query/key head dimension, denoted by `d_k` in the mathematical sections |
| `V` | - | Value head dimension, denoted by `d_v` in the mathematical sections |
| `q` | `[H, B, T, K]` | Query |
| `k` | `[H, B, T, K]` | Key |
| `v` | `[H, B, T, V]` | Value |
| `g` | `[H, B, T, K]` | Raw gate or an already activated natural-log decay |
| `beta` | `[H, B, T]` | Per-token delta-rule write coefficient |
| `o` | `[H, B, T, V]` | Token output |
| `final_state` | `[B, N, H, K, V]`, or `None` | Final `K x V` state for each sequence; `N=1` for fixed-length input |

### 1.2 Problem Addressed

Standard causal attention explicitly computes pairwise token relationships, so its representation of history grows with sequence length. KDA compresses history into a state matrix. Queries read from that state, keys and values update it through the delta rule, and a per-channel gate controls the forgetting rate of each key channel.

A direct token-by-token recurrence has the following temporal dependency:

```text
S_0 -> S_1 -> S_2 -> ... -> S_T
```

This form maps directly to the mathematical definition but provides little parallelism along the time dimension for long sequences. The current implementation partitions the sequence into fixed-size chunks:

1. Within a chunk, sequential dependencies are rewritten as a unit lower-triangular linear system whose dominant work is matrix multiplication.
2. Across chunks, only the fixed-size state matrix is propagated recurrently.
3. Variable-length sequences are aligned to chunk boundaries so that one chunk never crosses two independent sequences.
4. Context Parallelism (CP) connects portions of the same logical sequence on different ranks through affine state summaries.

### 1.3 Semantic Differences from Standard Attention

| Aspect | Standard causal attention | KDA chunk forward |
|---|---|---|
| History representation | Token-token attention scores | `d_k x d_v` state matrix |
| Time complexity | Typically grows with `T^2` | State path grows linearly with `T` |
| Normalization | Softmax | No softmax; memory is controlled by decay gates and the delta rule |
| Causality | Specified by an attention mask | Enforced jointly by state recurrence and lower-triangular intra-chunk matrices |
| Cross-chunk dependency | Usually no recurrent state | The final state of one chunk initializes the next chunk |
| Variable-length isolation | Masking or packed attention | Segment alignment, state reset, and chunk-to-sequence mapping |

KDA output is not a softmax-weighted sum. It is the sum of a read from the state that existed before the current chunk and the contributions written earlier within the current chunk. Together, these terms reproduce the token-level KDA recurrence.

## 2. Overall Design

### 2.1 End-to-End Flow

```text
PallasTpuKimiDeltaAttention._fwd
  |
  |-- Validate dtype, chunk size, and fixed-length divisibility
  |-- Residual path:    chunk_kda_fwd_custom
  |-- No-residual path: _chunk_kda_fwd_no_residuals
  |     -> CP/segment metadata, optional q/k L2 normalization,
  |        and variable-length alignment
  v
chunk_kda_fwd
  |
  |-- Stage 1+2: kda_fwd_intra_fused
  |     Gate activation/cumsum + intra-chunk triangular solve
  |     -> g_cumsum, w, u, kg, Aqk, Akk, optional qg
  |
  |-- CP bridge, when CP is active
  |     chunk_gated_delta_rule_fwd_h_pre_process
  |     -> all_gather_into_tensor -> _merge_initial_state
  |
  |-- Stage 3+4
  |     chunk_kda_fwd_h_o_varlen
  |       Variable length: aligned segment boundaries and chunk mapping
  |       Fixed length: one [0, T] segment per batch element
  v
Restore variable-length output layout and cast the output dtype
  |
  v
o, optional final_state
```

### 2.2 Four Semantic Stages

| Stage | Objective | Primary results |
|---|---|---|
| Stage 1 | Activate the gate and compute the cumulative decay within each chunk | `g_cumsum` |
| Stage 2 | Eliminate intra-chunk delta-rule sequential dependencies | `w`, `u`, `kg`, `Aqk`, `Akk` |
| Stage 3 | Propagate the state along the chunk dimension | Chunk-start states, `v_new`, and an optional final state |
| Stage 4 | Combine the history read with current-chunk contributions | `o` |

These stages describe mathematical dependencies, not four mandatory kernel dispatches. The current bfloat16 production path fuses Stages 1 and 2. Both fixed-length and variable-length execution fuse Stages 3 and 4 in the same kernel. The CP bridge sits between Stages 2 and 3 and reconstructs the correct initial state for the current rank.

### 2.3 Why the Computation Is Staged

1. Every decay ratio in Stage 2 depends on the chunk-local cumulative gate produced by Stage 1.
2. Stage 3 can propagate state only after Stage 2 has compressed intra-chunk updates into the `W/U` representation.
3. Stage 4 depends both on the chunk-start state from Stage 3 and on the intra-chunk relation matrix from Stage 2.
4. CP can form a state summary only after local `W/U` values are available, and it must finish the cross-rank merge before Stage 3 begins.

Keeping these semantic stages makes correctness easier to explain. Fusing adjacent stages reduces round trips to High Bandwidth Memory (HBM).

## 3. Primary Call Chain

### 3.1 Backend and Residual-Producing Calls

`PallasTpuKimiDeltaAttention._fwd` is the Tokamax TPU backend hook. It enforces bfloat16 or float32 input, `chunk_size=64`, and fixed-length divisibility by the chunk size. It then selects the residual-producing path or the no-residual path according to `return_residuals`.

The residual-producing path invokes `chunk_kda_fwd_custom`. This function normalizes the recurrent-state representation, prepares CP and variable-length metadata, applies optional query/key L2 normalization, aligns variable-length tensors once, and returns the aligned forward residual set when requested. When `scale` is not supplied, this path uses `K^{-1/2}`. It calls `chunk_kda_fwd` with `_skip_align=True` to prevent the same data from being aligned twice.

The no-residual path invokes `_chunk_kda_fwd_no_residuals`. It prepares CP and cumulative-length metadata, then lets `chunk_kda_fwd` perform any required normalization and alignment. Both paths ultimately enter `chunk_kda_fwd` and use the same KDA mathematical pipeline from that point onward.

### 3.2 Primary Functions and Responsibilities

| Function | Primary responsibility | Downstream function |
|---|---|---|
| `PallasTpuKimiDeltaAttention._fwd` | Tokamax TPU backend dispatch, dtype/chunk validation, and residual-path selection | `chunk_kda_fwd_custom` or `_chunk_kda_fwd_no_residuals` |
| `_chunk_kda_fwd_no_residuals` | Forward execution without materializing the optional forward residual set | `chunk_kda_fwd` |
| `chunk_kda_fwd_custom` | Residual-producing forward, one-time variable-length alignment, normalization, and residual preservation | `chunk_kda_fwd` |
| `chunk_kda_fwd` | Overall forward orchestration and selection of Stage 1+2, CP, and Stage 3+4 paths | Stage functions below |
| `kda_fwd_intra_fused` | Stage 1+2 dtype routing and fused execution | Fused bfloat16 path or exact float32 path |
| `chunk_gated_delta_rule_fwd_h_pre_process` | Construct a rank-local affine state summary for CP | `all_gather_into_tensor` |
| `_merge_initial_state` | Merge upstream rank summaries and recover the first local segment's initial state | Stage 3+4 |
| `chunk_kda_fwd_h_o_varlen` | Unified fused Stage 3+4 for fixed-length and variable-length input | `o`, optional state residual, and final state |

### 3.3 Single-Alignment Rule for Variable-Length Calls

When variable-length input is described by `segment_ids` with shape `[B,T]`, the entry path first produces a statically shaped `cu_seqlens`. Each segment is padded to a multiple of `chunk_size`, and a chunk-to-sequence mapping is generated.

The residual-producing forward performs this alignment before preserving residuals, so the lower-level forward consumes already aligned tensors. The no-residual path performs the same alignment inside `chunk_kda_fwd`. In either case, each call aligns its data exactly once. This guarantees that:

- Forward outputs and returned residual tensors use the same token mapping.
- `Aqk`, `Akk`, cumulative gates, and chunk indices remain consistent.
- Output is restored to the original token order and length only at the end of the call chain.

## 4. Mathematical Formulation

### 4.1 Symbols and Dimensions

The following derivation considers one batch element and one attention head, omitting batch and head indices.

| Symbol | Dimension | Meaning |
|---|---:|---|
| `L` | Scalar | Length of one sequence |
| `C` | Scalar | Chunk size; 64 on the current production path |
| `N_C` | Scalar | Number of chunks: `L/C`, or aligned length divided by `C` |
| `d_k` | Scalar | Query/key dimension, corresponding to backend `K` |
| `d_v` | Scalar | Value dimension, corresponding to backend `V` |
| `q_t, k_t` | `R^{d_k}` | Query/key column vectors at token `t` |
| `v_t, o_t` | `R^{d_v}` | Value/output column vectors at token `t` |
| `alpha_t` | `(0,1]^{d_k}` | Per-key-channel decay factors |
| `beta_t` | `R` | Delta-rule write coefficient |
| `S_t` | `R^{d_k x d_v}` | State after token `t` has been processed |
| `Q_c, K_c` | `R^{C x d_k}` | Stacked query/key matrices for chunk `c` |
| `V_c, O_c` | `R^{C x d_v}` | Stacked value/output matrices for chunk `c` |
| `S_c^0` | `R^{d_k x d_v}` | State at the start of chunk `c` |

`Diag(x)` denotes a matrix with vector `x` on its diagonal. `Tril(X)` and `StrictTril(X)` retain the lower-triangular and strictly lower-triangular parts, respectively. `odot` denotes elementwise multiplication with broadcasting, and `oslash` denotes elementwise division.

### 4.2 Token-Level KDA Recurrence: Paper Eq. (1)

The KDA recurrence in the paper is:

$$
S_t = \left(I-\beta_t k_t k_t^\top\right)
      \operatorname{Diag}(\alpha_t)S_{t-1}
      + \beta_t k_t v_t^\top,
\qquad
o_t = S_t^\top q_t.
$$

To represent the output scale used by the implementation, let `s` be the attention scale and define:

$$
\widetilde q_t = s q_t.
$$

The implementation therefore computes `S_t^T \widetilde q_t`. This is identical to the paper when `s=1`.

The token-level state update can also be written in delta-correction form:

$$
\bar S_t = \operatorname{Diag}(\alpha_t)S_{t-1},
$$

$$
e_t = v_t - \bar S_t^\top k_t,
$$

$$
S_t = \bar S_t + \beta_t k_t e_t^\top.
$$

`e_t` is the reconstruction error of the current state's mapping from `k_t` to `v_t`. The state first decays channel by channel and is then corrected through a rank-1 update. The gate controls memory lifetime, while the delta rule reduces interference between old and new associations.

### 4.3 Cumulative Decay Within a Chunk

For local position `r in [1,C]` in chunk `c`, define the cumulative decay used by the paper:

$$
\gamma_c^{i\rightarrow j}
= \prod_{m=i}^{j}\alpha_c^m,
\qquad
\gamma_c^r = \gamma_c^{1\rightarrow r}.
$$

The matrix formulation more often uses the ratio between two cumulative decays:

$$
\rho_c^{i\rightarrow r}
= \gamma_c^r \oslash \gamma_c^i
= \prod_{m=i+1}^{r}\alpha_c^m,
\qquad i \le r.
$$

This ratio is the per-channel decay experienced by a write at position `i` before position `r` is reached. Define:

$$
\Gamma_c =
\begin{bmatrix}
(\gamma_c^1)^\top\\
\vdots\\
(\gamma_c^C)^\top
\end{bmatrix}
\in \mathbb{R}^{C\times d_k}.
$$

The implementation does not multiply `alpha` values directly. Instead, it defines `ell_t = ln(alpha_t)` and computes the following value independently in each chunk:

$$
G_c^r = \log_2\gamma_c^r
= \frac{1}{\ln 2}\sum_{m=1}^{r}\ell_c^m.
$$

Consequently:

$$
\gamma_c^r = 2^{G_c^r},
\qquad
\rho_c^{i\rightarrow r}=2^{G_c^r-G_c^i}.
$$

When `use_gate_in_kernel=True`, the input is a raw gate. Let `x_t = g_t^{raw} + dt_bias` and `lambda_h = exp(A_log_h)`. The implementation supports two parameterizations of `ell_t`:

$$
\ell_t = -\lambda_h\,\operatorname{softplus}(x_t),
$$

or, when a negative `lower_bound = a` is provided:

$$
\ell_t = a\,\operatorname{sigmoid}(\lambda_h x_t).
$$

When `use_gate_in_kernel=False`, input `g` directly represents `ell=ln(alpha)`. Production semantics require this value to be non-positive so that `alpha` lies in `(0,1]`.

### 4.4 Chunk Expansion and Compact WY Representation: Paper Eq. (2)-(5)

Define the token-level state transition matrix:

$$
T_c^r =
\left(I-\beta_c^r k_c^r(k_c^r)^\top\right)
\operatorname{Diag}(\alpha_c^r).
$$

Partially expanding the token-level recurrence within one chunk gives Paper Eq. (2):

$$
S_c^r =
\underbrace{\left(\prod_{m=1}^{r}T_c^m\right)}_{P_c^r} S_c^0
+
\underbrace{\sum_{i=1}^{r}
\left(\prod_{m=i+1}^{r}T_c^m\right)
\beta_c^i k_c^i(v_c^i)^\top}_{H_c^r}.
$$

Matrix products act from right to left in temporal order, so later transitions appear on the left. The compact WY representation compresses a sequence of rank-1 transformations into a small set of matrix factors. For KDA, it can be written as:

$$
P_c^r = \operatorname{Diag}(\gamma_c^r)
-\sum_{i=1}^{r}
\operatorname{Diag}(\rho_c^{i\rightarrow r})
k_c^i(w_c^i)^\top,
$$

$$
H_c^r = \sum_{i=1}^{r}
\operatorname{Diag}(\rho_c^{i\rightarrow r})
k_c^i(u_c^i)^\top.
$$

The auxiliary vectors follow Paper Eq. (4)-(5):

$$
w_c^r = \beta_c^r\left[
\operatorname{Diag}(\gamma_c^r)k_c^r
-\sum_{i=1}^{r-1}w_c^i
\left((k_c^i)^\top
\operatorname{Diag}(\rho_c^{i\rightarrow r})k_c^r\right)
\right],
$$

$$
u_c^r = \beta_c^r\left[
v_c^r
-\sum_{i=1}^{r-1}u_c^i
\left((k_c^i)^\top
\operatorname{Diag}(\rho_c^{i\rightarrow r})k_c^r\right)
\right].
$$

These equations still contain a recurrence over positions. The lower-triangular matrix transform in Paper Eq. (6) rewrites them as a linear system suitable for matrix units.

### 4.5 Lower-Triangular Transform and Stage 2 Outputs: Paper Eq. (6)-(7)

For one chunk, define:

$$
K_c^+ = \Gamma_c\odot K_c,
\qquad
K_c^- = K_c\oslash\Gamma_c,
\qquad
D_{\beta,c}=\operatorname{Diag}(\beta_c).
$$

The strictly lower-triangular dependency matrix is:

$$
L_c = \operatorname{StrictTril}
\left(D_{\beta,c}K_c^+(K_c^-)^\top\right)
\in \mathbb{R}^{C\times C}.
$$

Paper Eq. (6) defines:

$$
M_c = (I+L_c)^{-1}D_{\beta,c}.
$$

Paper Eq. (7) gives:

$$
W_c = M_cK_c^+ \in \mathbb{R}^{C\times d_k},
\qquad
U_c = M_cV_c \in \mathbb{R}^{C\times d_v}.
$$

The implementation first computes and stores:

$$
A_c=(I+L_c)^{-1},
$$

then evaluates the equivalent expressions:

$$
W_c=A_c(D_{\beta,c}K_c^+),
\qquad
U_c=A_c(D_{\beta,c}V_c).
$$

The runtime tensor `Akk` denotes `A_c`; it is not the unsolved key-key relation matrix. Runtime tensors `w` and `u` denote `W_c` and `U_c`, respectively.

Stage 2 also constructs the scaled causal query-key matrix:

$$
A_{qk,c}=
\operatorname{Tril}\left[
(\Gamma_c\odot \widetilde Q_c)(K_c^-)^\top
\right]
\in \mathbb{R}^{C\times C},
$$

and the gated key required to propagate writes to the end of the chunk:

$$
\bar K_c^r = \rho_c^{r\rightarrow C}\odot k_c^r
= k_c^r\odot 2^{G_c^C-G_c^r}.
$$

Runtime tensor `Aqk` denotes `A_{qk,c}`, while `kg` denotes the stacked `bar K_c`. `Aqk` already includes the attention scale and the causal lower-triangular constraint.

### 4.6 Chunk State Update and Output: Paper Eq. (8)-(9)

Given chunk-start state `S_c^0`, first compute:

$$
V_c^{new}=U_c-W_cS_c^0
\in \mathbb{R}^{C\times d_v}.
$$

Paper Eq. (8) gives the final state of the chunk:

$$
S_{c+1}^0
= \operatorname{Diag}(\gamma_c^C)S_c^0
+\bar K_c^\top V_c^{new}.
$$

Paper Eq. (9) gives all outputs in the chunk:

$$
O_c
= (\Gamma_c\odot\widetilde Q_c)S_c^0
+A_{qk,c}V_c^{new}.
$$

The first term reads history that existed before the current chunk and is the inter-chunk contribution. The second term contains only writes that have already occurred within the current chunk and is the intra-chunk contribution. The lower-triangular structure of `Aqk` prevents position `r` from reading a future position `i>r`.

### 4.7 Variable-Length Sequences

Let the original interval of segment `n` in a batch be:

$$
[p_n,p_{n+1}),
\qquad
L_n=p_{n+1}-p_n.
$$

Each segment independently executes the recurrence above:

$$
S_{n,0}=S_{n,init},
$$

and state never propagates between different segments. The implementation extends the execution length of each segment to:

$$
\widehat L_n=C\left\lceil\frac{L_n}{C}\right\rceil.
$$

Query, key, value, and beta are zero at padded positions. For a pre-activated gate, zero padding directly represents the identity decay `alpha=1`. When gate activation is performed in the fused kernel, the aligned raw-gate padding is replaced with `-1e4`; after either the softplus or lower-bound sigmoid activation, this yields an effectively zero log decay rather than the nonzero decay that raw zero padding would produce. Padded positions therefore produce no output and preserve the carried state. After execution, outputs are mapped back to the original intervals `[p_n,p_{n+1})`.

### 4.8 Affine State Summaries for Context Parallelism

Expanding Eq. (8) as an affine function of the chunk-start state gives:

$$
S_{c+1}^0=\Phi_c S_c^0+\Psi_c,
$$

where:

$$
\Phi_c=\operatorname{Diag}(\gamma_c^C)-\bar K_c^\top W_c
\in\mathbb{R}^{d_k\times d_k},
$$

$$
\Psi_c=\bar K_c^\top U_c
\in\mathbb{R}^{d_k\times d_v}.
$$

The composition of two consecutive summaries `a` and `b` remains affine:

$$
\Phi_{b\circ a}=\Phi_b\Phi_a,
\qquad
\Psi_{b\circ a}=\Phi_b\Psi_a+\Psi_b.
$$

Any number of consecutive chunks on one rank can therefore be compressed into fixed-size `(Phi_rank, Psi_rank)` tensors. Runtime CP summary `M` corresponds to `Phi_rank`, and `S_ext` corresponds to `Psi_rank`. This runtime `M` is distinct from the lower-triangular transform matrix `M_c` in Paper Eq. (6).

## 5. Current Physical Implementation

### 5.1 Stage 1+2: Gate Accumulation and Intra-Chunk Solve

#### 5.1.1 Objective, Inputs, and Outputs

`kda_fwd_intra_fused` dispatches Stages 1 and 2. The Tokamax backend contract and the Pallas kernels both use head-first `[H, B, T_a, ...]` tensors, where `T_a` is either the fixed input length or the aligned variable-length input length, and `N_T=T_a/C`.

Inputs:

| Tensor | Shape | Meaning |
|---|---:|---|
| `q`, `k` | `[H, B, T_a, K]` | Query/key |
| `v` | `[H, B, T_a, V]` | Value |
| `g` | `[H, B, T_a, K]` | Raw gate or `ln(alpha)` |
| `beta` | `[H, B, T_a]` | Write coefficient |
| `A_log` | `[H]` or `None` | Raw-gate activation parameter |
| `dt_bias` | `[H*K]` or `None` | Raw-gate bias |

Outputs:

| Tensor | Shape | Mathematical meaning |
|---|---:|---|
| `g_cumsum` | `[H, B, T_a, K]`, float32 | `G=log2(Gamma)` |
| `w` | `[H, B, T_a, K]` | `W` |
| `u` | `[H, B, T_a, V]` | `U` |
| `kg` | `[H, B, T_a, K]` | `bar K` |
| `Aqk` | `[H, B, T_a, C]` | Flattened storage of `A_qk` |
| `Akk` | `[H, B, T_a, C]` | Flattened storage of `(I+L)^{-1}` |
| `qg` | `[H, B, T_a, K]` or `None` | Optionally materialized `Gamma odot Q` |

#### 5.1.2 Gate Activation and Accumulation

The bfloat16 production path performs chunk-local accumulation inside `_fused_gate_intra_kernel` and also performs gate activation there when `use_gate_in_kernel=True`. It multiplies the activated or pre-activated gate by a fixed lower-triangular matrix of ones to produce the chunk-local prefix sum:

$$
G_c=\operatorname{Tril}(\mathbf 1_{C\times C})
\frac{\ell_c}{\ln 2}.
$$

`G_c` is consumed immediately to construct `L` and `Aqk`, without first being written to and reread from HBM. Because Stages 3+4 and the residual-producing forward path require the cumulative gate, `g_cumsum` is eventually written as a stage output.

#### 5.1.3 Relation Matrices and Numerical Reference Points

The current production path fixes `C=64` and uses relation-construction sub-block size `BC=16` to construct `Aqk` and `L`. Given a reference vector `mu` for a sub-block, the exponent difference is factored as:

$$
2^{G_r-G_i}=2^{G_r-\mu}\,2^{\mu-G_i}.
$$

When `safe_gate=True`, the midpoint position of the sub-block is used as the reference, reducing the exponent range on either side. Outside the causal region, the implementation sets exponent differences to zero before evaluating `exp2`, then zeros the corresponding matrix-multiplication inputs. This prevents masked positions from first producing infinity and then forming a non-finite result when multiplied by zero.

The query-key and key-key left operands for a sub-block are concatenated. One batched matrix multiplication then produces both the corresponding rows of `Aqk` and `L`, reusing gated-key loads and matrix-multiplication scheduling.

#### 5.1.4 Lower-Triangular Solve

The bfloat16 path uses inversion sub-block size `BC_inv=8` and splits `L` into a block-diagonal strictly lower-triangular part `L_d` and a cross-block part `F`:

$$
L=L_d+F.
$$

Each `8 x 8` diagonal block is strictly lower triangular, so `L_d^8=0` in exact arithmetic and:

$$
P=(I+L_d)^{-1}=\sum_{j=0}^{7}(-L_d)^j.
$$

Now define:

$$
G=PF,
\qquad
(I+L)^{-1}=(I+G)^{-1}P.
$$

`G` is also nilpotent in the block lower-triangular structure formed by the eight blocks, so `(I+G)^{-1}` is obtained from another finite Neumann expansion. The implementation organizes these finite series with doubling and replaces row-by-row forward substitution with small matrix multiplications.

In exact arithmetic, this expansion is an equivalent inverse of a strictly lower-triangular system; it is not a truncation of an infinite series. Differences between the bfloat16 path and row-wise solving come from input quantization and floating-point matrix-multiplication rounding. Float32 input takes the exact forward-substitution path for higher-precision execution and numerical comparison.

The same inverse is applied once to concatenated `D_beta V`, `D_beta K+`, and identity-matrix right-hand sides. This produces `U`, `W`, and `Akk` together and avoids repeatedly solving the same triangular system.

#### 5.1.5 TPU Benefits and Intermediate-State Policy

- Fusing Stages 1 and 2 removes one write-then-read round trip of `g_cumsum` through HBM.
- Relation sub-block size `BC=16` and inversion sub-block size `BC_inv=8` bound peak Vector Memory (VMEM) use.
- One program point can process multiple heads as a mini-batch, amortizing Direct Memory Access (DMA) and dispatch overhead.
- Relation construction and solving use float32 accumulators before outputs are converted to the input compute dtype.
- `Aqk` and `Akk` remain available in the forward result tuple. `g_cumsum` is retained when required by the selected forward residual policy. `w/u/kg/qg` are released when `disable_recompute=False`.

### 5.2 CP Bridge: State Merge Between Stages 2 and 3

#### 5.2.1 Objective, Inputs, and Outputs

Under CP, one logical sequence can continue from an upstream rank onto the current rank. Stages 1+2 can operate entirely on rank-local tokens, but Stage 3 requires the true initial state of the first local segment.

`chunk_gated_delta_rule_fwd_h_pre_process` accepts:

| Tensor | Shape |
|---|---:|
| `kg`, `w` | `[H, B, T_local_aligned, K]` |
| `u` | `[H, B, T_local_aligned, V]` |
| `g_cumsum` | `[H, B, T_local_aligned, K]` |
| `cu_seqlens` | `[N_local+1]` or `[B, N_local+1]` |

It processes only the final real local segment of each batch element and returns:

| Tensor | Shape | Meaning |
|---|---:|---|
| `S_ext_local` | `[H, B, K, V]`, float32 | `Psi_rank` assuming a zero input state |
| `M_local` | `[H, B, K, K]`, float32 | `Phi_rank` |

#### 5.2.2 Local Summary Computation

For each chunk in the final local segment, compute:

$$
\Phi_c=\operatorname{Diag}(2^{G_c^C})-\bar K_c^\top W_c,
$$

$$
\Psi_c=\bar K_c^\top U_c.
$$

Two float32 VMEM scratch buffers accumulate:

$$
S_{ext}\leftarrow\Phi_cS_{ext}+\Psi_c,
$$

$$
M\leftarrow\Phi_cM,
$$

starting from `S_ext=0` and `M=I`. The rank-local summary is written at the final chunk of the segment.

Only the final local segment is compressed because segments that start and end on one rank do not require cross-device state transfer. Only the trailing segment can continue onto the next rank.

#### 5.2.3 Cross-Rank Merge

`all_gather_into_tensor` collects `(S_ext_local, M_local)` from all ranks along the CP mesh axis. Based on the number of preceding ranks that belong to the same logical sequence, `_merge_initial_state` applies upstream affine summaries in order and reconstructs the input state for the current rank.

The recovered state is written to internal `initial_state[:, 0]`; all remaining local segment states remain zero. Only the first local segment can continue from an upstream rank. Later local segments begin on the current rank.

When `B>1`, each batch element may have different cross-rank boundaries, so summaries are merged independently per batch element. Communication and merge arithmetic remain in float32 to avoid amplifying state error through low-precision affine products across many ranks.

#### 5.2.4 Memory and Communication

Each rank communicates one `K x V` tensor `S_ext` and one `K x K` tensor `M`; the payload is independent of `T_local`. The current order is local preprocessing, all-gather, local merge, then Stage 3+4. The implementation does not explicitly overlap the all-gather with Stage 3+4 computation.

### 5.3 Stage 3+4: State Propagation and Output

#### 5.3.1 Objective, Inputs, and Outputs

Stages 3+4 consume the compressed results of Stages 1+2 and implement Paper Eq. (8)-(9). The primary inputs are:

| Tensor | Shape |
|---|---:|
| `w`, `kg` | `[H, B, T_a, K]` |
| `u` | `[H, B, T_a, V]` |
| `g_cumsum` | `[H, B, T_a, K]` |
| `q` | `[H, B, T_a, K]` |
| `Aqk` | `[H, B, T_a, C]` |
| `initial_state` | `[B,N,H,K,V]` at the fused kernel boundary; `N=1` for fixed-length input |

The primary outputs are:

| Tensor | Shape | Meaning |
|---|---:|---|
| `o` | `[H, B, T_a, V]` | Output in aligned internal layout |
| `h` | `[H, B, N_T, K, V]` or `None` | Optional saved state before each chunk |
| `final_state` | `[B,N,H,K,V]` or `None` at the fused kernel boundary | Final state of each sequence |

#### 5.3.2 Per-Chunk Computation

For chunk-start state `S_c`, the computation is ordered as follows:

$$
V_c^{new}=U_c-W_cS_c,
$$

$$
O_c^{inter}=(\Gamma_c\odot\widetilde Q_c)S_c,
$$

$$
O_c^{intra}=A_{qk,c}V_c^{new},
$$

$$
O_c=O_c^{inter}+O_c^{intra},
$$

$$
S_{c+1}=\operatorname{Diag}(\gamma_c^C)S_c
+\bar K_c^\top V_c^{new}.
$$

This order lets the same `V_new` feed both the intra-chunk output and the state update. State is initialized from the corresponding initial state at the first chunk of a sequence and is optionally written as the final state after the final chunk.

#### 5.3.3 Unified Fused Path for Fixed and Variable Length

Both fixed-length and variable-length input call `chunk_kda_fwd_h_o_varlen`. The running state remains in a float32 VMEM scratch buffer. Head-group and batch grid dimensions are parallel, while the chunk dimension is declared `arbitrary` to preserve its ordered dependency.

Each chunk performs four primary matrix multiplications:

1. `W @ S` to produce `V_new`.
2. Gated query `@ S` to produce the inter-chunk output.
3. `Aqk @ V_new` to produce the intra-chunk output.
4. `kg^T @ V_new` to update the state.

`V_new` is produced and consumed inside the current program point and is not written to HBM by the forward orchestrator. The running state remains in scratch across chunks. The pre-update state of each chunk is written only when `disable_recompute=True` requests `h`.

The inter-chunk output uses the first cumulative-gate row in the chunk, `G_ref`, as a reference point:

$$
(Q\odot2^G)S
=\left(Q\odot2^{G-G_{ref}}\right)
\left(2^{G_{ref}}\odot S\right).
$$

Both exponent terms are bounded below by `-126`, treating extremely small decays as stable underflow to zero. State decay is also computed in float32. At a segment start, scratch is reset or loaded from the initial state. At a segment end, the final state is written when requested.

#### 5.3.4 Fixed-Length Encoding and State Contract

Fixed-length input is represented for the unified kernel as one logical segment `[0,T]` per batch element. The orchestrator constructs `cu_seqlens` with shape `[B,2]` and a chunk mapping with shape `[B,T/C,2]`; every mapping entry identifies segment zero and its physical chunk index. This reuses the variable-length kernel's boundary handling and ordered state recurrence without variable-length gathering, padding, or output unalignment.

The Tokamax backend recurrent-state contract is `[B,N,H,K,V]`, with `N=1` for fixed-length input. Inside `chunk_kda_fwd`, the fixed-length state is temporarily represented as `[B,H,K,V]` and a singleton segment dimension is inserted before the fused call. The fused kernel returns `[B,1,H,K,V]`; the lower orchestrator removes that dimension for its fixed-length return, and the backend applies `as_public_final_state` before returning to Tokamax.

The orchestrator always sets `store_v_new=False`, so `V_new` is consumed within the kernel rather than spilled to HBM. It sets `store_h=disable_recompute`, writing the per-chunk state only under the forward save-state policy.

#### 5.3.5 Recompute and Caching

`disable_recompute` controls forward intermediate materialization without changing `o` or `final_state`:

- With `disable_recompute=True`, Stage 1+2 materializes `qg`, Stage 3+4 writes per-chunk `h`, and `chunk_kda_fwd` leaves `w`, `u`, `qg`, `kg`, and `h` in its result tuple. `V_new` is still not stored because the orchestrator fixes `store_v_new=False`.
- With `disable_recompute=False`, `chunk_kda_fwd` clears `w`, `u`, `qg`, `kg`, `V_new`, and `h` after producing the forward output. When raw-gate activation runs in the fused kernel, it also clears `g_cumsum`; `Aqk` and `Akk` remain available.

CP and non-CP execution use the same materialization policy. The switch does not alter the CP bridge equations for `(S_ext, M)`, the initial state recovered for the current rank, or the numerical forward result.

## 6. TPU-Side Optimizations

### 6.1 Fusion Boundaries

The implementation selects two high-value fusion boundaries:

1. **Stage 1+2 fusion:** The cumulative gate directly participates in constructing `L`, `Aqk`, and `W/U` while still in VMEM, reducing HBM round trips.
2. **Unified Stage 3+4 fusion:** For both fixed-length and variable-length input, the running state and `V_new` are produced, consumed, and updated in one program point, without materializing token-level `V_new`.

The four stages are not fused into one kernel because a CP communication boundary exists between Stages 2 and 3, and the residual-producing forward contract keeps Stage 2 tensors such as `Aqk/Akk` available after Stage 1+2. The current fusion scope balances forward HBM traffic, residual materialization, and implementation complexity.

### 6.2 Data Layout, Tiling, and Mini-Batching

- The Tokamax backend and the main kernels use head-first `[H,B,T,...]` layout throughout, making a head or head group a primary Pallas grid dimension without an entry-layout transpose.
- The complete delivered path fixes `C=64`, making chunk counts, masks, lower-triangular solve steps, and block specifications compile-time constants.
- Stage 1+2 uses `BC=16` relation sub-blocks and `BC_inv=8` inversion sub-blocks for the finite Neumann expansion.
- Multiple heads can form a mini-batch within one program point for batched matrix multiplication. The mini-batch is capped at 16 and adjusted to divide `H`.
- Stage 3+4 pads `K` and `V` to TPU major-block alignment, trading a small amount of padded computation for contiguous DMA and stable matrix-multiplication tiles.
- The current state-propagation and CP-summary paths require `K<=256`, keeping `K x V` state and `K x K` transition matrices within the intended VMEM budget.

### 6.3 VMEM and HBM Data Flow

The primary stage boundaries are:

```text
Stage 1+2 HBM outputs
  g_cumsum, w, u, kg, Aqk, Akk, optional qg

Optional CP HBM/collective boundary
  S_ext, M -> all-gather -> initial_state

Stage 3+4 outputs
  o, optional final_state, optional h
```

The key policies are:

- The cumulative gate first remains in VMEM inside the fused kernel, then is written once for downstream stages.
- The Stage 3+4 running state for both fixed-length and variable-length input remains in a float32 VMEM scratch buffer.
- `V_new` is not written to HBM by the forward orchestrator.
- `h` is written only when `disable_recompute=True` selects the save-state path.
- `W/U/kg` are cleared from the lower forward result when `disable_recompute=False`.

The current implementation still transfers `w/u/kg/Aqk/g_cumsum` across the Stage 1+2 to Stage 3+4 boundary. This is the primary remaining HBM boundary in forward.

### 6.4 Matrix-Multiplication-Oriented Reformulation

TPU matrix units favor regular dense matrix multiplications with static shapes over scalar updates or row-wise operations on long dependency chains. The implementation therefore uses the following reformulations:

- Prefix sum becomes multiplication by a fixed lower-triangular matrix.
- The WY auxiliary-vector recurrence becomes the lower-triangular system from Paper Eq. (6).
- Lower-triangular inversion becomes finite Neumann expansions organized by doubling.
- Query-key and key-key sub-blocks are combined into one batched matrix multiplication.
- `W @ S`, `Aqk @ V_new`, and `kg^T @ V_new` use the same chunk tile.

These transformations add some small-matrix intermediate work but substantially reduce temporal control dependencies and increase matrix-unit utilization.

### 6.5 Work Control for Variable-Length Input

Each segment is padded only to its own nearest chunk boundary. If one batch element contains `N` segments, total per-segment tail padding satisfies:

$$
0\le\sum_{n=1}^{N}(\widehat L_n-L_n)<N(C-1).
$$

This avoids padding every segment to the longest sequence in the batch. For `B>1`, each batch element is aligned independently by segment and is then padded to one common static total length to satisfy JAX compilation shape requirements.

`chunk_indices` maps physical chunks to logical segments. Stages 3+4 reset state at segment starts and write state at segment ends. CP preprocessing handles only the final real segment and skips zero-length placeholder segments introduced by `N_max`.

### 6.6 Computation and Communication

Non-CP forward performs no cross-device communication. CP executes in the following order:

```text
Rank-local Stage 1+2
  -> rank-local (S_ext, M)
  -> all-gather float32 summaries
  -> rank-local initial-state merge
  -> rank-local Stage 3+4
```

Communication volume is independent of sequence length. Each batch/head/rank sends only one `K x V` summary and one `K x K` summary. The current implementation focuses on reducing communication volume and does not explicitly overlap communication with computation.

### 6.7 Precision and Numerical Stability

- Gate activation, cumulative gates, state, CP summaries, and critical matrix-multiplication accumulators use float32.
- Bfloat16 is the production dtype for fused Stage 1+2. Float32 uses separate cumulative-gate computation and exact lower-triangular forward substitution.
- All decays use `exp2` in log2 space, avoiding repeated conversion between natural and binary exponential representations.
- The `safe_gate` reference point reduces exponent ranges in Stage 2 sub-blocks.
- Anti-causal entries are cleared before `exp2`, preventing overflow at positions that will be masked.
- The unified Stage 3+4 path factors `2^G` around a reference point and applies a `-126` lower bound to very small exponents.
- CP summaries and multi-rank merging remain in float32.
- Final `o` is cast back to the query input dtype.

## 7. Supported Functionality

| Capability | Delivered behavior | Call-chain location and applicability |
|---|---|---|
| Fixed-length forward | Supports batches, multiple heads, and optional initial/final state | `T` is partitioned with `chunk_size=64`, represented as one segment per batch element, and passed to `chunk_kda_fwd_h_o_varlen` |
| Variable-length forward | Supports `segment_ids` shaped `[B,T]` | The backend constructs `cu_seqlens`, aligns data, then calls `chunk_kda_fwd_h_o_varlen` |
| Segments not aligned to 64 | Supported | Each segment is padded independently to a multiple of 64, then output is restored to the original length |
| Variable length with `B>1` | Supported | Each batch element independently constructs boundaries, alignment, and chunk mapping |
| Context Parallel forward | Supports rank-local `segment_ids` and a minimal `CPContext` | Rank-chain metadata is derived automatically; delivered CP form uses no external initial state and does not request final state |
| CP with `B>1` | Supported | Each batch element independently merges upstream state summaries |
| Causal semantics | Supported | `Aqk` is lower triangular and `L` is strictly lower triangular |
| Segment boundaries | Supported | Chunks do not cross segments; Stages 3+4 reset state at boundaries |
| Padding semantics | Supported | Data is zero padded; pre-activated gate padding is zero, while raw-gate padding is replaced with `-1e4` before fused activation |
| Raw-gate activation | Supports softplus decay and negative-lower-bound sigmoid decay | `kda_fwd_intra_fused`; requires the corresponding `A_log`, with optional `dt_bias` |
| Pre-activated gate | Supported | `g` represents `ln(alpha)` and is non-positive under production semantics |
| Query/key L2 normalization | Supported | The residual-producing helper or lower forward orchestrator normalizes before Pallas computation |
| Initial state | Tokamax backend contract `[B,N,H,K,V]`; fixed length uses `N=1` | Loaded at the first chunk of each sequence |
| Final state | Tokamax backend contract `[B,N,H,K,V]`; supported for non-CP fixed-length and variable-length execution | Written at the final chunk of each sequence |
| bfloat16 | Supported | Fused Stage 1+2 with float32 critical accumulators |
| float32 | Supported | Separate gate cumsum and exact lower-triangular forward substitution |
| Recompute control | Supported | `disable_recompute` controls preservation of chunk-start state `h` |
| Static segment bound | Supports `N_max` | Fixes compile-time shapes for `cu_seqlens` and chunk mapping; a tight value reduces empty-segment overhead |

### 7.1 Mask and Causal Semantics

Three layers jointly implement the current forward masking semantics:

1. `Tril` on `Aqk` ensures that an output position reads only the current and earlier positions.
2. `StrictTril` on `L` ensures that a position's delta correction depends only on earlier writes.
3. Segment alignment and state reset isolate logical sequences, while padded positions preserve zero writes and identity decay; raw-gate mode uses `-1e4` padding so fused gate activation approaches that identity behavior.

Causality, segment boundaries, and padding are therefore fixed components of the KDA recurrence rather than three independently supplied arbitrary masks.

### 7.2 Forward Semantics of `disable_recompute`

The flag changes only the forward intermediate set:

- `disable_recompute=True`: Forward preserves per-chunk `h` and leaves the Stage 1+2 intermediates selected by `chunk_kda_fwd` materialized. HBM use is higher.
- `disable_recompute=False`: Forward releases `h`, `w`, `u`, `qg`, `kg`, and `V_new` after producing `o` and the optional final state. HBM use is lower.

Both modes retain `Aqk` and `Akk`. When gate activation runs in the fused kernel, `g_cumsum` is also released in the `disable_recompute=False` mode. Neither mode changes forward output semantics.

## 8. Design Trade-Offs and Boundaries

### 8.1 Trade-Offs in the Current Decomposition

A fully token-recurrent implementation has the most direct mathematical form but low temporal parallelism. Fully expanding the entire sequence would forfeit the linear-complexity advantage of fixed-size state. The current design takes an intermediate approach:

- Within a chunk, the compact WY representation and lower-triangular system convert recurrence into matrix computation.
- Across chunks, state recurrence is retained.
- Adjacent stages are fused where doing so does not break the CP boundary or the forward residual-output contract.
- Variable-length input is aligned to obtain regular tiles.
- CP uses affine summaries instead of transferring token-level state.

### 8.2 Performance-Oriented Choices

- `chunk_size=64` fixes the tuning target and compilation shapes of the complete forward path.
- The bfloat16 path uses finite Neumann/doubling matrix chains to reduce sequential row-wise solving.
- Stage 1+2 and the unified fixed-length/variable-length Stage 3+4 path use fused kernels to reduce HBM intermediates.
- `K/V` alignment and head mini-batching can add a small amount of padded computation but improve DMA continuity and matrix-unit utilization.
- `disable_recompute` exposes a configurable trade-off between forward HBM usage and retained intermediate state.

### 8.3 Required Runtime Invariants

- Production gate semantics require `alpha in (0,1]`, so pre-activated `ln(alpha)` is non-positive.
- `lower_bound=None` selects the softplus gate; a non-`None` value selects the sigmoid gate variant. These forward entry points pass the value through without imposing an additional numeric range check.
- The complete production path uses `chunk_size=64`.
- Fixed-length `T` must be divisible by the chunk size; the call chain automatically aligns each segment for variable-length input.
- State propagation and CP summary paths require `K<=256`.
- Positive `segment_ids` identify valid segments, while `0` denotes padding.
- `N_max` is a compile-time upper bound on segment count. It must be no smaller than the real segment count and should be as tight as practical.
- CP boundaries are derived from rank-local `segment_ids` and the CP mesh axis. Only the first local segment can inherit state from an upstream rank.
- Causal, segment-boundary, and padding semantics must remain consistent with both the lower-triangular system and state-reset logic; changing only one layer is incorrect.

## 9. Performance Measurements

### 9.1 Environment

| Item | Value |
|------|-------|
| Chips | 2x2x1 (4-chip TPU v7) |
| JAX | 0.10.2 |
| libtpu | 0.0.42.1 |
| TPU device | 8 TpuDevice (4 chips x 2 cores, topology 2x2x1) |
| dtype | bf16 |
| Shape | B=1, T=8192, H=32, K=128, V=128, C=64, N_SEQS=25 (25 variable-length sequences packing T=8192) |

The fused-pipeline kernels (`1_intra_fused`, `2_fused_h_o`, `3_e2e_fused`) correspond to Stage 1+2, the H·O output stage, and the full fused four-stage path respectively. The CP column compares the single-rank baseline against the `cp_size=4` CP path, where the sequence is split along the CP mesh axis into four rank-local segments of length 2048.

### 9.2 Non-CP Fused Pipeline (cp off)

| Kernel | Median (ms) | Min | P90 |
|--------|------------:|----:|----:|
| `1_intra_fused` | 1.548 | 1.529 | 1.563 |
| `2_fused_h_o` | 1.180 | 1.166 | 1.191 |
| `3_e2e_fused` | 3.944 | 3.922 | 3.972 |

`2_fused_h_o` is the fastest stage (1.180 ms), 0.76x of `1_intra_fused` (1.548 ms) and 0.30x of `3_e2e_fused` (3.944 ms). The H·O stage benefits the most from fusion because its read/write pattern is direct: it consumes the chunk-start state and chunk-local correlation matrices produced by Stage 1+2 and produces the token output, with no further sequential dependency. `3_e2e_fused` is the heaviest because it concatenates the full Stage 1->2->3->4 chain in one dispatch, so it absorbs the recurrence/scan cost of all four stages.

### 9.3 Context Parallel (cp on, cp_size=4)

Measurement uses the same varlen input (Section 9.1), with `cp_size=4` splitting `T=8192` into four rank-local segments of 2048 along the CP mesh axis. Single-rank `baseline_chunk_kda` is the non-CP baseline run on the same input.

| Kernel | Median (ms) | Min | P90 |
|--------|------------:|----:|----:|
| `baseline_chunk_kda` (single-rank baseline) | 3.515 | 3.493 | 3.544 |
| `cp_e2e` (CP path, cp_size=4) | 2.288 | 2.246 | 2.352 |

**CP scaling:**

```text
baseline_chunk_kda     3.540 ms   (cp off, single rank)
cp_e2e                 2.322 ms   (cp on,  cp_size=4, 4 ranks)
speedup = 1.52x (ideal 4x), efficiency = 38.1%
```

### 9.4 Where CP Cost Comes From

The non-CP and CP core computation is identical; the CP path adds one extra step between Stage 2 and Stage 3 -- the rank-local affine state summary and its cross-rank merge (Section 5.2).

From profiling, the main cost of CP is not the collective communication itself. Consistent with Section 6.6, `all_gather` only exchanges a compressed `K x V` and `K x K` summary per batch/head/rank, independent of sequence length, so it is relatively cheap. The real cost is the local pre-merge before communication:

- `chunk_gated_delta_rule_fwd_h_pre_process` must accumulate the running state and state-transition matrices along this rank's chunk sequence to produce the affine summary;
- this step has a scan/reduction nature (it walks the chunk sequence in order and folds every chunk's `W/U` update into the summary), so it cannot be fully parallelized the way a per-chunk matmul can;
- as a result, although CP eliminates the global sequential dependency, this rank-local scan is a residual sequential section that caps the achievable speedup well below the 4x ideal.

### 9.5 Optimization Focus

- **Non-CP path:** `3_e2e_fused` (3.944 ms) dominates end-to-end and carries the full four-stage recurrence/scan, making it the primary optimization target for single-rank throughput. Reducing repeated full-chunk reads and improving the parallelism of the inter-chunk state chain product are the levers.
- **CP path:** the bottleneck is the pre-merge scan in `chunk_gated_delta_rule_fwd_h_pre_process`, not the collective. CP efficiency (38.1% of ideal) lifts once this summary scan is addressed -- by reducing repeated full-chunk input reads, improving the chain-product/scan parallelism of the summary, or fusing part of the state-merge logic forward into an existing forward stage.

### 9.6 HBM Roofline (Non-CP, single rank)

Measurement: TPU **v7**, bf16, varlen 25 sequences, B=1, T=8192 (padded 9792), H=32, K=V=128, BT=64, NT=153. HBM bandwidth taken as 3.69 TB/s.

| Kernel | Runtime (us) | HBM (MB) | BW lower bound (us) | BW utilization |
|--------|-------------:|---------:|--------------------:|---------------:|
| `pallas_kda_fwd_intra_fused` | 932.3 | 886.0 | 240.1 | 25.8% |
| `chunk_kda_fwd_h_o_varlen` | 709.7 | 733.9 | 198.9 | 28.0% |
| `copy_bitcast_fusion` | 178.2 | 229.5 | 62.2 | 34.9% |
| `convert_bitcast_fusion` | 77.9 | 229.5 | 62.2 | 79.9% |

HBM volume comes from the LLO dump (the custom-call profiler reports `bytes_accessed=0` for these kernels); for `chunk_kda_fwd_h_o_varlen` the trace `bytes_accessed=769572864` (~734 MB) matches the LLO-derived `733.9 MB`. BW lower bound = HBM volume / peak HBM bandwidth; BW utilization = lower bound / runtime.

### 9.7 HBM Roofline (CP, per-device, cp_size=4)

Measurement: TPU **v7x-4** (cp_size=4, 4-device SPMD), bf16, varlen 25 segments each pre-aligned to BT (no `_align_seqs` padding), B=1, global T=8192, global NT=128, H=32, K=V=128, BT=64; sharded along the CP axis, per-rank T=8192/4=2048, NT_per_device=32. HBM bandwidth taken as 3.69 TB/s. Per-device values.

| Kernel | Runtime (us) | HBM (MB) | BW lower bound (us) | BW utilization |
|--------|-------------:|---------:|--------------------:|---------------:|
| `pallas_kda_fwd_intra_fused` | 354.1 | 221.5 | 60.0 | 16.9% |
| `chunk_kda_fwd_h_o_varlen` | 263.2 | 183.5 | 49.7 | 18.9% |
| CP `all-gather` | 45.4 | 10.0 | 2.7 | 6.0% |

The two core kernels are the same as the non-CP path; per-device runtime drops roughly 4x (intra: 932.3 -> 354.1 us; h_o: 709.7 -> 263.2 us), tracking the per-rank sequence length cut. The CP `all-gather` is cheap (45.4 us) and only exchanges the compressed `S_ext`+`M` summary, not full token tensors (Section 6.6); its low BW utilization reflects that it is latency/scan-bound rather than bandwidth-bound.
