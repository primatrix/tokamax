"""Utilities used by the experimental KDA implementation."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import os

import jax
import jax.numpy as jnp
from tokamax._src.ops.experimental.kda.cp_utils import (
    CPContext,
    _derive_cp_metadata_from_segment_ids,
)


def exp(x):
  return jnp.exp(x.astype(jnp.float32))


def exp2(x):
  """Base-2 exponential, matching Triton's tl.exp2."""
  return jnp.exp2(x.astype(jnp.float32))


def get_interpret() -> bool:
  env = os.environ.get("PALLAS_INTERPRET", "")
  return env.strip().lower() in ("1", "true")


def cdiv(x, y: int):
  return (x + y - 1) // y


def l2norm_fwd(x: jax.Array, eps: float = 1e-6):
  x_f = x.astype(jnp.float32)
  rstd = jax.lax.rsqrt(jnp.sum(x_f * x_f, axis=-1) + eps)
  return (x_f * rstd[..., None]).astype(x.dtype), rstd.astype(jnp.float32)


def l2norm_bwd(y: jax.Array, rstd: jax.Array, dy: jax.Array):
  y_f = y.astype(jnp.float32)
  dy_f = dy.astype(jnp.float32)
  rstd_f = rstd.astype(jnp.float32)
  dot_dy_y = jnp.sum(dy_f * y_f, axis=-1)
  dx = dy_f * rstd_f[..., None] - dot_dy_y[..., None] * y_f * rstd_f[..., None]
  return dx.astype(y.dtype)

def normalize_initial_state(
    initial_state: jax.Array | None,
    *,
    batch: int,
    heads: int,
    key_dim: int,
    value_dim: int,
) -> jax.Array | None:
  if initial_state is None:
    return None
  if initial_state.ndim != 5:
    raise ValueError(
        "`initial_state` must have shape [B, N, H, K, V]; got "
        f"{initial_state.shape}."
    )
  if initial_state.shape[0] != batch:
    raise ValueError(
        f"`initial_state` batch dimension must be {batch}; got {initial_state.shape}."
    )
  if initial_state.shape[2:] != (heads, key_dim, value_dim):
    raise ValueError(
        "`initial_state` trailing dimensions must be "
        f"{(heads, key_dim, value_dim)}; got {initial_state.shape[2:]}."
    )
  return initial_state


def as_public_final_state(
    final_state: jax.Array | None,
    *,
    segment_ids: jax.Array | None,
) -> jax.Array | None:
  if final_state is None:
    return None
  if final_state.ndim == 4 and segment_ids is None:
    return final_state[:, None]
  return final_state


def derive_cp_context(
    *,
    q: jax.Array,
    segment_ids: jax.Array | None,
    initial_state: jax.Array | None,
    output_final_state: bool,
    cp_context: CPContext | None,
    chunk_size: int,
    N_max: int | None,
) -> tuple[CPContext | None, jax.Array | None]:
  cu_seqlens = None
  if cp_context is None or not cp_context.is_cp_enabled:
    return cp_context, cu_seqlens

  if initial_state is not None:
    raise ValueError("`initial_state` is not supported when CP is enabled.")
  if output_final_state:
    raise ValueError("`output_final_state` is not supported when CP is enabled.")
  if segment_ids is None:
    raise ValueError("CP requires rank-local `segment_ids` with shape [B, T].")

  n_max = N_max if N_max is not None else cdiv(q.shape[2], chunk_size)
  cu_locals, chain_metas = [], []
  for b in range(segment_ids.shape[0]):
    cu_b, meta_b = _derive_cp_metadata_from_segment_ids(
        segment_ids[b],
        cp_context.axis_name,
        n_max=n_max,
    )
    cu_locals.append(cu_b)
    chain_metas.append(meta_b)
  cu_seqlens = jnp.stack(cu_locals, axis=0)
  chain_meta = {k: jnp.stack([m[k] for m in chain_metas]) for k in chain_metas[0]}
  cp_context = dataclasses.replace(
      cp_context,
      is_first_rank=chain_meta["is_first_rank"],
      is_last_rank=chain_meta["is_last_rank"],
      pre_num_ranks=chain_meta["pre_num_ranks"],
      post_num_ranks=chain_meta["post_num_ranks"],
  )
  return cp_context, cu_seqlens


def segment_ids_to_cu_seqlens(
    segment_ids: jax.Array | None,
    *,
    initial_state: jax.Array | None,
    chunk_size: int,
    N_max: int | None,
    seq_len: int,
) -> tuple[jax.Array | None, int | None]:
  if segment_ids is None:
    return None, N_max
  if N_max is None:
    N_max = (
        initial_state.shape[1]
        if initial_state is not None
        else cdiv(seq_len, chunk_size)
    )
  return segment_ids_to_seqlens(segment_ids, max_segs=N_max), N_max


def align_up(x, align: int):
  return cdiv(x, align) * align


def pad_to_multiple(x: jax.Array, multiple: int | list[int], axis: int | list[int], val):
  if isinstance(multiple, int):
    multiple = [multiple]
  if isinstance(axis, int):
    axis = [axis]
  if len(multiple) != len(axis):
    raise ValueError(
        f"Length of multiple {len(multiple)} must match axis {len(axis)}."
    )

  shape = list(x.shape)
  pad_width = [(0, 0)] * len(shape)
  for ax, mu in zip(axis, multiple):
    remainder = shape[ax] % mu
    if remainder:
      pad_width[ax] = (0, mu - remainder)
  return jnp.pad(x, pad_width, constant_values=val)


def prepare_lens(cu_seqlens: jax.Array) -> jax.Array:
  return cu_seqlens[1:] - cu_seqlens[:-1]


def compute_padded_cu_seqlens(cu_seqlens: jax.Array, chunk_size: int):
  """Round each sequence length up to `chunk_size` in a cu_seqlens array."""
  if cu_seqlens.ndim == 2:
    rows = [compute_padded_cu_seqlens(cu_seqlens[b], chunk_size) for b in range(cu_seqlens.shape[0])]
    return jnp.stack(rows, axis=0)
  lens = jnp.diff(cu_seqlens)
  padded_lens = cdiv(lens, chunk_size) * chunk_size
  return jnp.concatenate([
      jnp.zeros(1, dtype=cu_seqlens.dtype),
      jnp.cumsum(padded_lens),
  ])


def align_segment_ids(
    segment_ids: jax.Array,
    N_max: int,
    chunk_size: int,
) -> jax.Array:
  """Align 1D segment IDs to chunk boundaries, matching `_align_seqs`."""
  if segment_ids.ndim != 1:
    raise ValueError(f"`segment_ids` must be 1D, got {segment_ids.shape}.")
  T = segment_ids.shape[0]

  is_boundary = jnp.concatenate([
      jnp.ones(1, dtype=jnp.bool_),
      segment_ids[1:] != segment_ids[:-1],
  ])
  seg_idx = jnp.cumsum(is_boundary.astype(jnp.int32)) - 1
  positions = jnp.arange(T, dtype=jnp.int32)

  seg_starts = jnp.full(N_max, T, dtype=jnp.int32).at[seg_idx].min(positions)
  seg_ends = jnp.zeros(N_max, dtype=jnp.int32).at[seg_idx].max(positions + 1)
  seg_lens = jnp.maximum(seg_ends - seg_starts, 0)
  seg_labels = segment_ids[jnp.minimum(seg_starts, T - 1)]

  is_real = (seg_labels > 0) & (seg_lens > 0)
  aligned_lens = jnp.where(is_real, align_up(seg_lens, chunk_size), 0)
  aligned_starts = jnp.concatenate([
      jnp.zeros(1, dtype=jnp.int32),
      jnp.cumsum(aligned_lens)[:-1],
  ])

  T_aligned = ((T + N_max * (chunk_size - 1) + chunk_size - 1) // chunk_size) * chunk_size
  out = jnp.zeros((T_aligned,), dtype=jnp.int32)
  apos = jnp.arange(T_aligned, dtype=jnp.int32)

  def body(i, ids):
    start = aligned_starts[i]
    end = start + aligned_lens[i]
    mask = is_real[i] & (apos >= start) & (apos < end)
    return jnp.where(mask, seg_labels[i], ids)

  return jax.lax.fori_loop(0, N_max, body, out)


def segment_ids_to_seqlens(
    segment_ids: jax.Array,
    max_segs: int,
    chunk_size: int = 1,
) -> jax.Array:
  """Convert 1-indexed segment IDs with 0 padding to FLA-style cu_seqlens."""
  if segment_ids.ndim == 2:
    rows = [
        segment_ids_to_seqlens(segment_ids[b], max_segs, chunk_size)
        for b in range(segment_ids.shape[0])
    ]
    return jnp.stack(rows, axis=0)
  if segment_ids.ndim != 1:
    raise ValueError(f"`segment_ids` must be [T] or [B, T], got {segment_ids.shape}.")

  seg = segment_ids.reshape(-1)
  valid = seg != 0
  is_boundary = jnp.concatenate([
      jnp.ones(1, dtype=jnp.bool_),
      seg[1:] != seg[:-1],
  ]) & valid
  seg_idx = jnp.where(valid, jnp.cumsum(is_boundary.astype(jnp.int32)), 0)
  n_segs = seg_idx.max()
  n_real = jnp.sum(valid).astype(jnp.int32)

  is_end = jnp.concatenate([
      seg[1:] != seg[:-1],
      jnp.ones(1, dtype=jnp.bool_),
  ]) & valid
  prefix_len = jnp.cumsum(valid.astype(jnp.int32))

  drop_idx = jnp.asarray(max_segs + 1, dtype=jnp.int32)
  scatter_idx = jnp.where(is_end, seg_idx, drop_idx)
  scatter_val = jnp.where(is_end, prefix_len, 0)

  cu_seqlens = jnp.zeros((max_segs + 1,), dtype=jnp.int32)
  cu_seqlens = cu_seqlens.at[scatter_idx].max(scatter_val, mode="drop")
  out_idx = jnp.arange(max_segs + 1, dtype=jnp.int32)
  return jnp.where(out_idx > n_segs, n_real, cu_seqlens)


def prepare_chunk_indices(
    cu_seqlens: jax.Array,
    chunk_size: int,
    max_T: int | None = None,
) -> jax.Array:
  """Compute per-chunk `(seq_id, block_id)` mapping from cu_seqlens."""
  if cu_seqlens.ndim == 2:
    rows = [
        prepare_chunk_indices(cu_seqlens[b], chunk_size, max_T=max_T)
        for b in range(cu_seqlens.shape[0])
    ]
    return jnp.stack(rows, axis=0)
  lens = prepare_lens(cu_seqlens)
  n_chunks = cdiv(lens, chunk_size)
  num_seqs = len(lens)
  if max_T is None:
    max_T = cu_seqlens[-1]
  total_nt = max_T // chunk_size
  seq_ids = jnp.repeat(
      jnp.arange(num_seqs, dtype=jnp.int32),
      n_chunks,
      total_repeat_length=total_nt,
  )
  prefix_chunks = jnp.concatenate([
      jnp.zeros(1, dtype=jnp.int32),
      jnp.cumsum(n_chunks),
  ])
  seq_offsets = jnp.repeat(
      prefix_chunks[:-1],
      n_chunks,
      total_repeat_length=total_nt,
  )
  block_ids = jnp.arange(total_nt, dtype=jnp.int32) - seq_offsets
  return jnp.stack([seq_ids, block_ids], axis=1)


def assert_shape_or_none(
    x: jax.Array | list[jax.Array | None] | tuple[jax.Array | None, ...] | None,
    expected_shape: list[int] | tuple[int, ...],
    name: str | list[str] | tuple[str, ...] = "tensor",
):
  if x is None:
    return
  if isinstance(x, (list, tuple)):
    has_names = isinstance(name, (list, tuple)) and len(name) == len(x)
    for i, tensor in enumerate(x):
      if tensor is not None:
        curr_name = name[i] if has_names else f"{name}_{i}"
        assert tensor.shape == expected_shape, (
            f"[{curr_name}] Expected shape {expected_shape}, got {tensor.shape}"
        )
    return
  assert x.shape == expected_shape, (
      f"[{name}] Expected shape {expected_shape}, got {x.shape}"
  )


def assert_shape(
    x: jax.Array | list[jax.Array] | tuple[jax.Array, ...],
    expected_shape: list[int] | tuple[int, ...],
    name: str | list[str] | tuple[str, ...] = "tensor",
):
  if isinstance(x, (list, tuple)):
    has_names = isinstance(name, (list, tuple)) and len(name) == len(x)
    for i, tensor in enumerate(x):
      curr_name = name[i] if has_names else f"{name}_{i}"
      assert tensor.shape == expected_shape, (
          f"[{curr_name}] Expected shape {expected_shape}, got {tensor.shape}"
      )
    return
  assert x.shape == expected_shape, (
      f"[{name}] Expected shape {expected_shape}, got {x.shape}"
  )


def export_public(current_globals):
  return [
      name
      for name, value in current_globals.items()
      if not name.startswith("_") and callable(value)
  ]


@dataclass(frozen=True)
class TpuConfig:
  generation: str
  vmem_per_core_bytes: int
  smem_per_core_bytes: int
  tflops_bf16_2d: float
  tflops_fp8_2d: float
  tflops_fp32_2d: float
  block_align_minor: int = 8
  block_align_major: int = 128
  hbm_bandwidth_gbps: float = 0.0
  frequency_ghz: float = 0.0
  tflops_fp32_1d: float = 0.0
  description: str = ""
  num_lanes: int = 128
  num_sublanes: int = 8
  mxu_column_size: int = 128
  cmem_per_core_bytes: int = 0
  hbm_per_core_bytes: int = 0
  mem_bw_bytes_per_second: int = 0
  tflops_int8_2d: float = 0.0
  tflops_int4_2d: float = 0.0

  @property
  def vmem_limit_bytes(self) -> int:
    return int(self.vmem_per_core_bytes * 0.9)

  @property
  def vmem_hw_limit_bytes(self) -> int:
    return int(self.vmem_per_core_bytes * 0.9)


TPU_V6E = TpuConfig(
    generation="v6e",
    vmem_per_core_bytes=128 * 1024 * 1024,
    smem_per_core_bytes=1024 * 1024,
    tflops_fp8_2d=920.0,
    tflops_bf16_2d=920.0,
    tflops_fp32_2d=460.0,
    hbm_bandwidth_gbps=1640.0,
    frequency_ghz=1.75,
    tflops_fp32_1d=7.168,
    description="TPU v6e (Trillium)",
    mxu_column_size=256,
    hbm_per_core_bytes=34_400_000_000,
    mem_bw_bytes_per_second=int(1.64e12),
    tflops_int8_2d=1840.0,
    tflops_int4_2d=3680.0,
)

TPU_V7 = TpuConfig(
    generation="v7",
    vmem_per_core_bytes=64 * 1024 * 1024,
    smem_per_core_bytes=1024 * 1024,
    tflops_fp8_2d=2300.0,
    tflops_bf16_2d=1155.0,
    tflops_fp32_2d=577.5,
    hbm_bandwidth_gbps=3700.0,
    frequency_ghz=2.2,
    tflops_fp32_1d=9.0112,
    description="TPU v7 (Ironwood) - 2 devices per chip, values are per-device",
    mxu_column_size=256,
    hbm_per_core_bytes=103_000_000_000,
    mem_bw_bytes_per_second=int(3.70e12),
)

_PRESETS: dict[str, TpuConfig] = {
    "v6e": TPU_V6E,
    "v7": TPU_V7,
}

_DEVICE_KIND_MAP: list[tuple[str, str]] = [
    ("v6 lite", "v6e"),
    ("v7 lite", "v7"),
    ("v7e", "v7"),
    ("v6e", "v6e"),
    ("v7", "v7"),
    ("v6", "v6e"),
]

_current_config: TpuConfig | None = None


def _detect_tpu_config() -> TpuConfig:
  try:
    devices = jax.devices()
  except RuntimeError:
    devices = []

  for device in devices:
    device_kind = device.device_kind.lower()
    for needle, key in _DEVICE_KIND_MAP:
      if needle in device_kind:
        return _PRESETS[key]
  return TPU_V6E


def set_tpu_config(config: TpuConfig) -> None:
  global _current_config
  _current_config = config


def get_tpu_config() -> TpuConfig:
  global _current_config
  if _current_config is None:
    _current_config = _detect_tpu_config()
  return _current_config
