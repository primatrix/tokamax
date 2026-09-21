# Copyright 2026 Primatrix Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0

"""Independent FP32 segmented-attention oracle for selected full-length heads.

This is an accuracy diagnostic, never part of timed kernel execution. Query
chunking bounds memory but retains every key for each row. It does not use
Splash residuals, BF16 probability casts or the candidate's custom backward.
"""

import jax
import jax.numpy as jnp


def fp32_attention_and_gradients(
    q, k, v, do, q_ids, kv_ids, *, block_q=512,
    mask_value=-jnp.inf, barrier_stages=False,
):
  """Returns output, natural-log LSE, dQ, dK, dV for one head in FP32."""
  if q.ndim != 2 or k.ndim != 2 or v.ndim != 2 or do.ndim != 2:
    raise ValueError("Expected one head with rank-2 arrays")
  if block_q <= 0 or q.shape[0] % block_q:
    raise ValueError("block_q must be positive and divide the query length")
  if q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
    raise ValueError("Incompatible query/key/value dimensions")
  if do.shape != (q.shape[0], v.shape[1]):
    raise ValueError("Output cotangent shape does not match attention output")
  if q_ids.shape != (q.shape[0],) or kv_ids.shape != (k.shape[0],):
    raise ValueError("Segment IDs must match query/key lengths")
  q, k, v, do = (x.astype(jnp.float32) for x in (q, k, v, do))
  scale = jnp.float32(q.shape[1] ** -0.5)
  initial = (
      jnp.zeros_like(do), jnp.zeros((q.shape[0],), jnp.float32),
      jnp.zeros_like(q), jnp.zeros_like(k), jnp.zeros_like(v),
  )

  def dot(a, b):
    return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)

  def stage(value):
    return jax.lax.optimization_barrier(value) if barrier_stages else value

  def block(index, carry):
    output, lse, dq, dk, dv = carry
    start = index * block_q
    qi = jax.lax.dynamic_slice_in_dim(q, start, block_q)
    doi = jax.lax.dynamic_slice_in_dim(do, start, block_q)
    ids = jax.lax.dynamic_slice_in_dim(q_ids, start, block_q)
    logits = stage(dot(qi, k.T) * scale)
    logits = jnp.where(ids[:, None] == kv_ids[None, :], logits, jnp.float32(mask_value))
    log_norm = jax.scipy.special.logsumexp(logits, axis=-1)
    p = stage(jax.nn.softmax(logits, axis=-1))
    oi = stage(dot(p, v))
    dp = stage(dot(doi, v.T))
    delta = stage(jnp.sum(doi * oi, axis=-1, keepdims=True))
    ds = stage(p * (dp - delta))
    dqi = dot(ds, k) * scale
    return (
        jax.lax.dynamic_update_slice(output, oi, (start, 0)),
        jax.lax.dynamic_update_slice(lse, log_norm, (start,)),
        jax.lax.dynamic_update_slice(dq, dqi, (start, 0)),
        dk + dot(ds.T, qi) * scale,
        dv + dot(p.T, doi),
    )

  return jax.lax.fori_loop(0, q.shape[0] // block_q, block, initial)


def require_finite_oracle(values, *, head):
  names = ("output", "logsumexp", "dq", "dk", "dv")
  invalid = [name for name, value in zip(names, values) if not bool(jnp.isfinite(value).all())]
  if invalid:
    raise FloatingPointError(f"Independent FP32 oracle is non-finite: head={head}, fields={invalid}")


@jax.jit
def _accuracy_statistics(actual, expected):
  a, b = actual.astype(jnp.float32), expected.astype(jnp.float32)
  difference = a - b
  square_error = jnp.sum(difference * difference)
  reference_energy = jnp.sum(b * b)
  return (
      jnp.all(jnp.isfinite(a)) & jnp.all(jnp.isfinite(b)),
      jnp.max(jnp.abs(difference)),
      jnp.sqrt(square_error / a.size),
      jnp.sqrt(square_error / jnp.maximum(reference_energy, jnp.float32(1e-30))),
  )


def accuracy_statistics(actual, expected):
  """Report error, without treating an unspecified tolerance as acceptance."""
  if actual.shape != expected.shape:
    raise ValueError(f"Shape mismatch: {actual.shape} versus {expected.shape}")
  finite, max_abs, rms_abs, relative_l2 = _accuracy_statistics(actual, expected)
  return dict(finite=bool(finite), max_abs=float(max_abs), rms_abs=float(rms_abs),
              relative_l2=float(relative_l2), elements=actual.size)
