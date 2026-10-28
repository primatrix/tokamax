"""Shared CP8 input construction for baseline and fused benchmarks."""

from __future__ import annotations

import hashlib
from typing import Iterable

import ml_dtypes
import numpy as np


FIXED_IDS_SHA256 = (
    "1666aff5fdd908b43633b165378d01eee1b70da2552e9fac656ce8fc7aa712e0"
)


def make_segment_ids(n: int, mode: str = "multimodal") -> np.ndarray:
  if mode == "single":
    return np.ones(n, dtype=np.int32)
  if mode != "multimodal" or n != 262144:
    raise ValueError("only CP8 production multimodal and single fixtures are supported")
  ids = np.zeros(n, dtype=np.int32)
  for start, real_length, segment_id in (
      (0, 39322, 1),
      (39328, 78644, 2),
      (117984, 117965, 3),
  ):
    ids[start:start + real_length] = segment_id
  digest = hashlib.sha256(ids.tobytes()).hexdigest()
  if digest != FIXED_IDS_SHA256:
    raise AssertionError(f"fixture SHA drift: {digest} != {FIXED_IDS_SHA256}")
  return ids


def make_sharded_arrays(jax, specs, shardings, devices, *, seed: int = 1234):
  """Make deterministic BF16 global arrays from the requested shardings."""
  arrays = []
  for operand, (spec, sharding) in enumerate(zip(specs, shardings)):
    parts = []
    for rank, device in enumerate(devices):
      index = sharding.addressable_devices_indices_map(spec.shape)[device]
      local_shape = tuple(
          spec.shape[dim] if isinstance(s, slice) and s.start is None and s.stop is None
          else (s.stop - s.start if isinstance(s, slice) else s)
          for dim, s in enumerate(index)
      )
      rng = np.random.default_rng(np.random.SeedSequence([seed, operand, rank]))
      values = (rng.random(local_shape, dtype=np.float32) * 2 - 1).astype(
          ml_dtypes.bfloat16
      )
      parts.append(jax.device_put(values, device))
    arrays.append(jax.make_array_from_single_device_arrays(spec.shape, sharding, parts))
  return arrays


def make_replicated_ids(jax, np_module, ids, sharding):
  n = ids.shape[0]
  return (
      jax.device_put(np_module.broadcast_to(ids[:, None], (n, 128)), sharding),
      jax.device_put(np_module.broadcast_to(ids[None, :], (8, n)), sharding),
      jax.device_put(np_module.broadcast_to(np_module.arange(n, dtype=np.int32)[:, None],
                                            (n, 128)), sharding),
  )
