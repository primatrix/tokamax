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
import math

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import numpy as np
import pytest
from tokamax._src import jaxtyping
from tokamax._src.ops.experimental.kda import api
from tokamax._src.ops.experimental.kda.cp_utils import CPContext


def _compute_ulp(
    x: np.ndarray, dtype: jax.typing.DTypeLike
) -> np.ndarray:
  """Computes one unit in the last place for `dtype` at each value in `x`."""
  finfo = jnp.finfo(dtype)
  mantissa_bits = round(-math.log2(float(finfo.eps)))
  min_ulp = float(finfo.tiny) * float(finfo.eps)

  abs_x = np.abs(x).astype(np.float64)
  _, exponent = np.frexp(abs_x)
  ulp = np.ldexp(1.0, exponent.astype(np.int32) - mantissa_bits - 1)
  ulp = np.maximum(ulp, min_ulp)
  return np.where(abs_x == 0, min_ulp, ulp)


def compare_tensor(
    name: str,
    expected: jax.Array | np.ndarray | None,
    actual: jax.Array | np.ndarray | None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    max_ulp: int = 1,
    dtype: jax.typing.DTypeLike = jnp.bfloat16,
    compare_dtype: jax.typing.DTypeLike = np.float64,
) -> bool:
  """Compares two tensors and prints focused numerical diagnostics."""
  if expected is None and actual is None:
    print(f"[{name}] Both are None. PASS.")
    return True
  if expected is None or actual is None:
    print(f"[{name}] One is None. FAIL.")
    return False

  expected_dtype = np.dtype(expected.dtype).name
  actual_dtype = np.dtype(actual.dtype).name
  if expected_dtype != actual_dtype:
    print(
        f"[{name}] Dtype mismatch: Left {expected.dtype} vs "
        f"Right {actual.dtype}. FAIL."
    )
    return False

  expected_np = np.asarray(expected).astype(compare_dtype)
  actual_np = np.asarray(actual).astype(compare_dtype)
  if expected_np.shape != actual_np.shape:
    print(
        f"[{name}] Shape mismatch: Left {expected_np.shape} vs "
        f"Right {actual_np.shape}. FAIL."
    )
    if expected_np.squeeze().shape != actual_np.squeeze().shape:
      return False
    expected_np = expected_np.squeeze()
    actual_np = actual_np.squeeze()
    print(f"  Comparing squeezed shape: {expected_np.shape}")

  diff = np.abs(expected_np - actual_np)
  max_diff = np.max(diff)
  max_value = np.max(np.abs(actual_np))
  max_relative_diff = np.max(diff / (np.abs(actual_np) + 1e-12))
  is_close = np.allclose(
      expected_np,
      actual_np,
      atol=atol,
      rtol=rtol,
      equal_nan=True,
  )

  if not is_close:
    ulp = _compute_ulp(
        np.maximum(np.abs(expected_np), np.abs(actual_np)), dtype
    )
    tolerance = np.maximum(
        atol + rtol * np.abs(actual_np), max_ulp * ulp
    )
    is_close = bool(np.all(diff <= tolerance))

  print(f"[{name}] {'PASS' if is_close else 'FAIL'}")
  print(f"  Max Value        : {max_value:.6e}")
  print(f"  Max Abs Diff     : {max_diff:.6e}")
  print(f"  Max Rel Diff     : {max_relative_diff:.6e}")

  if not is_close:
    error_ratio = diff / (tolerance + 1e-12)
    index = np.unravel_index(np.argmax(error_ratio), error_ratio.shape)
    print(f"  Max Mismatch details at index {index}:")
    print(f"    Left (expected) = {expected_np[index]}")
    print(f"    Right (actual)  = {actual_np[index]}")
    print(f"    Diff            = {diff[index]}")
    print(f"    Tolerance       = {tolerance[index]}")
    print(f"    ULP diff        = {diff[index] / ulp[index]:.2f}")
    print(f"    Ratio           = {error_ratio[index]}")

  return is_close


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
        name="fixed_unaligned_kv",
        seq_len=64,
        heads=1,
        key_dim=129,
        value_dim=127,
    ),
    _Case(
        name="cp2",
        seq_len=128,
        heads=2,
        seq_lens=(128,),
        cp_size=2,
        key_dim=128,
        value_dim=128,
        dtype=jnp.float32,
    ),
)


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

  segment_ids = (
      _make_segment_ids(case.seq_lens, real_seq_len)
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
      (case.heads, 1, real_seq_len, case.value_dim),
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


def _call_attention(
    implementation: api.Implementation,
    case: _Case,
    inputs: _Inputs,
    *,
    cp_context: CPContext | None = None,
):
  return api.kimi_delta_attention(
      inputs.q,
      inputs.k,
      inputs.v,
      inputs.g,
      inputs.beta,
      segment_ids=inputs.segment_ids,
      cp_context=cp_context,
      implementation=implementation,
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
    output, _ = _call_attention(
        "pallas_tpu",
        case,
        local_inputs,
        cp_context=cp_context,
    )
    return output

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
  return _call_attention(
      "xla" if reference else "pallas_tpu", case, inputs
  )


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

    def local_forward(q, k, v, g, beta):
      current_inputs = dataclasses.replace(
          local_inputs, q=q, k=k, v=v, g=g, beta=beta
      )
      output, _ = _call_attention(
          "pallas_tpu", case, current_inputs, cp_context=cp_context
      )
      return output

    _, pullback = jax.vjp(local_forward, q, k, v, g, beta)
    return pullback(dout)

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


def _direct_backward(
    implementation: api.Implementation, case: _Case, inputs: _Inputs
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
    output, final_state = _call_attention(
        implementation, case, current_inputs
    )
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

  for name, actual, expected in zip(
      ("output", "final_state"), output, reference, strict=True
  ):
    assert compare_tensor(
        name,
        expected,
        actual,
        atol=0.05,
        rtol=0.05,
        dtype=actual.dtype if actual is not None else case.dtype,
    ), f"{name} mismatch"


@pytest.mark.parametrize("case", _case_params())
def test_chunk_kda_backward(case: _Case):
  _require_tpu(case)
  inputs = _make_inputs(case)

  grads = (
      _cp_backward(case, inputs)
      if case.cp_size > 1
      else _direct_backward("pallas_tpu", case, inputs)
  )
  reference_grads = _direct_backward("xla", case, inputs)
  jax.block_until_ready((grads, reference_grads))

  grad_names = ["dq", "dk", "dv", "dg", "dbeta"]
  if case.use_initial_state:
    grad_names.append("dh0")
  for name, actual, expected in zip(
      grad_names, grads, reference_grads, strict=True
  ):
    assert compare_tensor(
        name,
        expected,
        actual,
        atol=0.05,
        rtol=0.05,
        dtype=actual.dtype,
    ), f"Gradient mismatch for {name}"


if __name__ == "__main__":
  pytest.main([__file__, "-v", "-s"])
