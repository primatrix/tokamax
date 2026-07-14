# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Numerical tests for the experimental Pallas TPU KDA implementation."""

import dataclasses

import chex
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.kda import base
from tokamax._src.ops.experimental.kda import pallas_tpu
from tokamax._src.ops.experimental.kda.cp_utils import CPContext


@dataclasses.dataclass(frozen=True)
class _Case:
  name: str
  seq_len: int
  heads: int
  seq_lens: tuple[int, ...] | None = None
  cp_size: int = 1
  key_dim: int = 128
  value_dim: int = 128
  dtype: jax.typing.DTypeLike = jnp.bfloat16
  use_initial_state: bool = False
  output_final_state: bool = False
  use_qk_l2norm_in_kernel: bool = False
  use_gate_in_kernel: bool = False
  safe_gate: bool = True
  lower_bound: float | None = None
  disable_recompute: bool = True
  long: bool = False

  @property
  def n_max(self) -> int | None:
    return len(self.seq_lens) if self.seq_lens is not None else None


@dataclasses.dataclass(frozen=True)
class _Inputs:
  q: jax.Array
  k: jax.Array
  v: jax.Array
  g: jax.Array
  beta: jax.Array
  A_log: jax.Array
  dt_bias: jax.Array
  initial_state: jax.Array | None
  segment_ids: jax.Array | None
  dout: jax.Array
  dfinal_state: jax.Array | None


_CASES = (
    _Case(
        name="fixed_t8192",
        seq_len=8192,
        heads=16,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        long=True,
    ),
    _Case(
        name="varlen",
        seq_len=256,
        heads=2,
        seq_lens=(45, 80, 20),
        use_initial_state=True,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=-0.01,
        disable_recompute=False,
    ),
    _Case(
        name="cp2",
        seq_len=128,
        heads=2,
        seq_lens=(128,),
        cp_size=2,
    ),
)

_PALLAS = pallas_tpu.PallasTpuKimiDeltaAttention()
_PALLAS_VJP = pallas_tpu.PallasTpuKimiDeltaAttentionVjp()
_REFERENCE = base.KimiDeltaAttention()


def _case_params():
  return [
      pytest.param(
          case,
          id=case.name,
          marks=(pytest.mark.long,) if case.long else (),
      )
      for case in _CASES
  ]


def _l2_normalize(x: jax.Array) -> jax.Array:
  x_f32 = x.astype(jnp.float32)
  rstd = jax.lax.rsqrt(jnp.sum(jnp.square(x_f32), axis=-1) + 1e-6)
  return (x_f32 * rstd[..., None]).astype(x.dtype)


def _make_segment_ids(seq_lens: tuple[int, ...], seq_len: int) -> jax.Array:
  ids = np.zeros((1, seq_len), dtype=np.int32)
  offset = 0
  for segment_id, length in enumerate(seq_lens, start=1):
    ids[0, offset : offset + length] = segment_id
    offset += length
  return jnp.asarray(ids)


def _pad_sequence(x: jax.Array, seq_len: int) -> jax.Array:
  pad_len = seq_len - x.shape[2]
  if x.ndim == 4:
    return jnp.pad(x, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
  return jnp.pad(x, ((0, 0), (0, 0), (0, pad_len)))


def _make_inputs(case: _Case) -> _Inputs:
  real_seq_len = (
      sum(case.seq_lens) if case.seq_lens is not None else case.seq_len
  )
  if real_seq_len > case.seq_len:
    raise ValueError(
        f"Real sequence length {real_seq_len} exceeds T={case.seq_len}."
    )
  if case.seq_len % case.cp_size != 0:
    raise ValueError("Global sequence length must be divisible by cp_size.")
  if case.seq_len // case.cp_size % 64 != 0:
    raise ValueError("Each device must receive a multiple of 64 tokens.")

  keys = jax.random.split(jax.random.key(42), 9)
  qk_shape = (case.heads, 1, real_seq_len, case.key_dim)
  v_shape = (case.heads, 1, real_seq_len, case.value_dim)

  q = jax.nn.silu(jax.random.normal(keys[0], qk_shape, dtype=jnp.float32))
  k = jax.nn.silu(jax.random.normal(keys[1], qk_shape, dtype=jnp.float32))
  q = q.astype(case.dtype)
  k = k.astype(case.dtype)
  if not case.use_qk_l2norm_in_kernel:
    q = _l2_normalize(q)
    k = _l2_normalize(k)

  v = jax.random.normal(keys[2], v_shape, dtype=jnp.float32).astype(
      case.dtype
  )
  g_raw = jax.random.normal(keys[3], qk_shape, dtype=jnp.float32)
  beta = jax.nn.sigmoid(
      jax.random.normal(
          keys[4], (case.heads, 1, real_seq_len), dtype=jnp.float32
      )
  ).astype(case.dtype)

  A_log = jnp.log(
      jax.random.uniform(
          keys[5], (case.heads,), minval=1.0, maxval=16.0
      )
  )
  dt = jnp.exp(
      jax.random.uniform(keys[6], (case.heads * case.key_dim,))
      * (jnp.log(jnp.array(0.1)) - jnp.log(jnp.array(0.001)))
      + jnp.log(jnp.array(0.001))
  ).clip(min=1e-4)
  dt_bias = dt + jnp.log(-jnp.expm1(-dt))

  if case.use_gate_in_kernel:
    g = g_raw.astype(case.dtype)
  else:
    gate_input = g_raw + dt_bias.reshape(
        case.heads, 1, 1, case.key_dim
    )
    g = (
        -jnp.exp(A_log).reshape(case.heads, 1, 1, 1)
        * jax.nn.softplus(gate_input)
    ).astype(case.dtype)

  q = _pad_sequence(q, case.seq_len)
  k = _pad_sequence(k, case.seq_len)
  v = _pad_sequence(v, case.seq_len)
  g = _pad_sequence(g, case.seq_len)
  beta = _pad_sequence(beta, case.seq_len)

  segment_ids = (
      _make_segment_ids(case.seq_lens, case.seq_len)
      if case.seq_lens is not None
      else None
  )
  state_count = case.n_max or 1
  initial_state = None
  if case.use_initial_state:
    initial_state = 0.1 * jax.random.normal(
        keys[7],
        (1, state_count, case.heads, case.key_dim, case.value_dim),
        dtype=jnp.float32,
    )

  dout = 0.1 * jax.random.normal(
      keys[8],
      (case.heads, 1, case.seq_len, case.value_dim),
      dtype=jnp.float32,
  )
  dfinal_state = None
  if case.output_final_state:
    dfinal_state = jnp.full(
        (1, state_count, case.heads, case.key_dim, case.value_dim),
        0.01,
        dtype=jnp.float32,
    )

  return _Inputs(
      q=q,
      k=k,
      v=v,
      g=g,
      beta=beta,
      A_log=A_log,
      dt_bias=dt_bias,
      initial_state=initial_state,
      segment_ids=segment_ids,
      dout=dout,
      dfinal_state=dfinal_state,
  )


def _attention_kwargs(case: _Case, inputs: _Inputs) -> dict[str, object]:
  return dict(
      A_log=inputs.A_log if case.use_gate_in_kernel else None,
      dt_bias=inputs.dt_bias if case.use_gate_in_kernel else None,
      scale=case.key_dim**-0.5,
      initial_state=inputs.initial_state,
      output_final_state=case.output_final_state,
      use_qk_l2norm_in_kernel=case.use_qk_l2norm_in_kernel,
      use_gate_in_kernel=case.use_gate_in_kernel,
      safe_gate=case.safe_gate,
      lower_bound=case.lower_bound,
      disable_recompute=case.disable_recompute,
      chunk_size=64,
      N_max=case.n_max,
  )


def _call_forward(
    op: base.KimiDeltaAttention,
    case: _Case,
    inputs: _Inputs,
    *,
    cp_context: CPContext | None = None,
    return_residuals: bool = False,
):
  return op._fwd(  # pylint: disable=protected-access
      inputs.q,
      inputs.k,
      inputs.v,
      inputs.g,
      inputs.beta,
      segment_ids=inputs.segment_ids,
      cp_context=cp_context,
      return_residuals=return_residuals,
      config=None,
      **_attention_kwargs(case, inputs),
  )


def _cp_mesh(case: _Case) -> tuple[Mesh, CPContext]:
  devices = jax.devices()[: case.cp_size]
  mesh = Mesh(np.asarray(devices), ("context",))
  return mesh, CPContext(mesh=mesh, axis_name="context")


def _cp_forward(case: _Case, inputs: _Inputs) -> tuple[jax.Array, None]:
  mesh, cp_context = _cp_mesh(case)

  def local_forward(q, k, v, g, beta, segment_ids):
    local_inputs = dataclasses.replace(
        inputs,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        segment_ids=segment_ids,
    )
    output, _ = _call_forward(
        _PALLAS,
        case,
        local_inputs,
        cp_context=cp_context,
    )
    return output[0]

  # The CP pre-process accepts batched cu_seqlens, but its runtime annotation
  # currently describes only the unbatched fast path.
  with jaxtyping.disable_jaxtyping(), jax.set_mesh(mesh):
    forward = jax.jit(
        jax.shard_map(
            local_forward,
            mesh=mesh,
            in_specs=(P(None, None, "context", None),) * 4
            + (P(None, None, "context"), P(None, "context")),
            out_specs=P(None, None, "context", None),
            check_vma=False,
        )
    )
    output = forward(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.g,
        inputs.beta,
        inputs.segment_ids,
    )
  return output, None


def _forward(case: _Case, inputs: _Inputs, *, reference: bool = False):
  if not reference and case.cp_size > 1:
    return _cp_forward(case, inputs)
  output, _ = _call_forward(
      _REFERENCE if reference else _PALLAS, case, inputs
  )
  return output


def _select_input_grads(
    case: _Case, grads: dict[str, jax.Array]
) -> tuple[jax.Array, ...]:
  names = ("q", "k", "v", "g", "beta")
  if case.use_initial_state:
    names += ("initial_state",)
  return tuple(grads[name] for name in names)


def _pallas_backward(
    case: _Case,
    inputs: _Inputs,
    *,
    cp_context: CPContext | None = None,
) -> tuple[jax.Array, ...]:
  output, residuals = _call_forward(
      _PALLAS,
      case,
      inputs,
      cp_context=cp_context,
      return_residuals=True,
  )
  if residuals is None:
    raise ValueError("Pallas forward did not return backward residuals.")
  grads, _ = _PALLAS_VJP._fwd(  # pylint: disable=protected-access
      residuals,
      output,
      (inputs.dout, inputs.dfinal_state),
      inputs.q,
      inputs.k,
      inputs.v,
      inputs.g,
      inputs.beta,
      segment_ids=inputs.segment_ids,
      cp_context=cp_context,
      return_residuals=True,
      config=None,
      **_attention_kwargs(case, inputs),
  )
  return _select_input_grads(case, grads)


def _cp_backward(case: _Case, inputs: _Inputs) -> tuple[jax.Array, ...]:
  mesh, cp_context = _cp_mesh(case)

  def local_backward(q, k, v, g, beta, segment_ids, dout):
    local_inputs = dataclasses.replace(
        inputs,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        segment_ids=segment_ids,
        dout=dout,
    )
    return _pallas_backward(case, local_inputs, cp_context=cp_context)

  qkv_spec = P(None, None, "context", None)
  beta_spec = P(None, None, "context")
  with jaxtyping.disable_jaxtyping(), jax.set_mesh(mesh):
    backward = jax.jit(
        jax.shard_map(
            local_backward,
            mesh=mesh,
            in_specs=(qkv_spec,) * 4
            + (beta_spec, P(None, "context"), qkv_spec),
            out_specs=(qkv_spec,) * 4 + (beta_spec,),
            check_vma=False,
        )
    )
    return backward(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.g,
        inputs.beta,
        inputs.segment_ids,
        inputs.dout,
    )


def _reference_backward(
    case: _Case, inputs: _Inputs
) -> tuple[jax.Array, ...]:
  def loss_fn(q, k, v, g, beta, initial_state):
    current_inputs = dataclasses.replace(
        inputs,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
    )
    output, final_state = _forward(case, current_inputs, reference=True)
    loss = jnp.sum(output.astype(jnp.float32) * inputs.dout)
    if inputs.dfinal_state is not None:
      if final_state is None:
        raise ValueError("Expected a final state for the backward case.")
      loss += jnp.sum(final_state * inputs.dfinal_state)
    return loss

  argnums = (
      (0, 1, 2, 3, 4, 5)
      if case.use_initial_state
      else (0, 1, 2, 3, 4)
  )
  return jax.grad(loss_fn, argnums=argnums)(
      inputs.q,
      inputs.k,
      inputs.v,
      inputs.g,
      inputs.beta,
      inputs.initial_state,
  )


def _require_tpu(case: _Case) -> None:
  if jax.default_backend() != "tpu":
    pytest.skip("Pallas TPU KDA tests require a TPU backend.")
  if len(jax.devices()) < case.cp_size:
    pytest.skip(f"Case requires {case.cp_size} TPU devices.")


@pytest.mark.parametrize("case", _case_params())
def test_chunk_kda_forward(case: _Case):
  _require_tpu(case)
  inputs = _make_inputs(case)

  output = _forward(case, inputs)
  reference = _forward(case, inputs, reference=True)
  jax.block_until_ready((output, reference))

  chex.assert_trees_all_close(output, reference, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("case", _case_params())
def test_chunk_kda_backward(case: _Case):
  _require_tpu(case)
  inputs = _make_inputs(case)

  grads = (
      _cp_backward(case, inputs)
      if case.cp_size > 1
      else _pallas_backward(case, inputs)
  )
  reference_grads = _reference_backward(case, inputs)
  jax.block_until_ready((grads, reference_grads))

  chex.assert_trees_all_close(
      grads, reference_grads, atol=0.08, rtol=0.08
  )


if __name__ == "__main__":
  pytest.main([__file__, "-v", "-s"])
