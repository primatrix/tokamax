# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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

"""MLA backward with tile-ready dK/dV and dQ reduction/export.

Derived from the C14/P18 numerically qualified CP8/CP16 allwrite candidate.
Arithmetic is delegated to the pinned Tokamax inner kernel. The caller is
vendored as ordinary source so production never rewrites or executes strings.
Keep the upstream source hash check when changing Tokamax: these Ref positions
and partial-buffer lifetimes are an internal ABI, not a public API.
"""
from __future__ import annotations

import functools
import contextlib
import hashlib
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as sk

SOURCE_SHA = "d798fdd38edd5aed5650344b72bd43724cf4345a4c9e0a92fb722a9e6ccd81f9"

MaskFunctionType = sk.MaskFunctionType
MaskInfo = sk.MaskInfo
NUM_LANES = sk.NUM_LANES
NUM_SUBLANES = sk.NUM_SUBLANES
QKVLayout = sk.QKVLayout
SplashConfig = sk.SplashConfig
_bytes = sk._bytes
_div = sk._div
_group_segment_spec = sk._group_segment_spec
_use_native_layout = sk._use_native_layout
from_head_minor = sk.from_head_minor
functools = sk.functools
get_kernel_name = sk.get_kernel_name
json = sk.json
lax = sk.lax
mask_info_lib = sk.mask_info_lib
to_i32 = sk.to_i32


def validate_metadata(kernel, cp):
  """Bind early-finalization and prefetch safety to actual static metadata."""
  info = kernel.dkv_mask_info
  count = np.asarray(info.num_active_blocks)
  raw_rows, raw_cols = np.asarray(info.active_rows), np.asarray(info.active_cols)
  if (
      cp not in (8, 16)
      or count.size != 1
      or not np.issubdtype(count.dtype, np.integer)
      or raw_rows.ndim != 1
      or raw_cols.ndim != 1
      or raw_rows.shape != raw_cols.shape
      or not np.issubdtype(raw_rows.dtype, np.integer)
      or not np.issubdtype(raw_cols.dtype, np.integer)
  ):
    raise ValueError("MLA export requires one static integer causal schedule and CP8/CP16")
  n = int(count.reshape(-1)[0])
  if not 0 < n <= raw_rows.size:
    raise ValueError("MLA active program count is outside the static schedule")
  rows, cols = raw_rows[:n], raw_cols[:n]
  kv_steps = len(np.unique(rows))
  q_steps = int(cols.max()) + 1
  if q_steps != kv_steps * 8 or kv_steps % cp:
    raise ValueError("MLA export requires eight Q tiles per KV tile and complete CP partitions")
  expected = [(j, i) for j in range(kv_steps) for i in range(8 * j, q_steps)]
  if list(zip(rows.tolist(), cols.tolist())) != expected:
    raise ValueError("MLA export requires the complete ordered causal tile schedule")
  # Mirror the pinned find_bounds end predicate on the *whole* metadata
  # array, including padding outside the dynamic grid. Interior Q programs
  # do not write the BF16 stage. Each KV tile must instead finalize exactly
  # once, in order, so wait_send(ordinal - 1) always matches a launched send.
  # A repeated final row in padding would suppress the last active write.
  write_end = raw_rows != np.roll(raw_rows, -1)
  write_end[-1] = True
  written_tiles = rows[write_end[:n]]
  if not np.array_equal(written_tiles, np.arange(kv_steps)):
    raise ValueError("MLA export requires exactly one final write per consecutive KV tile")
  last, distances = {}, []
  for g, (j, i) in enumerate(expected):
    key = (j % 3, i)
    if key in last:
      distances.append(g - last[key])
    last[key] = g
  # Other partials read by diagonal finalization must be older than the
  # two outstanding output generations, independently of same-slot reuse.
  previous, final_read_distances = {}, []
  for g, (j, i) in enumerate(expected):
    if j == i // 8:
      for partial in range(3):
        if partial != j % 3 and (partial, i) in previous:
          final_read_distances.append(g - previous[partial, i])
    previous[j % 3, i] = g
  if final_read_distances and min(final_read_distances) <= 2:
    raise ValueError("MLA final reduction can read an outstanding partial store")
  # Same-address lookahead must not race an outstanding partial store.
  if distances and min(distances) <= 2:
    raise ValueError("MLA dQ partial buffers can revisit an outstanding store")
  return dict(
      programs_per_head=n,
      kv_steps=kv_steps,
      q_steps=q_steps,
      export_writes_per_head=len(written_tiles),
      minimum_partial_revisit_distance=min(distances) if distances else None,
      minimum_final_other_partial_distance=min(final_read_distances) if final_read_distances else None,
  )


def dkv_output_kernel(*refs, cp, axis_name, cp_barrier, depth, serial, heads, kv_steps, **params):
  """One grid invocation; delegate all dQ/dK/dV arithmetic unchanged.

  Source (rank c, head h, KV tile j) uniquely owns destination
  (rank j//T, head c*H+h, sequence tile j%T), T=kv_steps/CP.
  Each destination tile has its own receive semaphore for this invocation.
  Send slots may be reused only after wait_send; all receives drain at exit.
  """
  assert len(refs) == 29
  original = list(refs[:25])
  stage_k, stage_v, send_sem, recv_sem = refs[25:]
  out_k, out_v = original[20:22]
  assert original[17] is original[18] is None  # No GQA dKV aliases.
  assert not params["native_layout"]
  bkv = params["bkv"]
  tiles_per_peer = kv_steps // cp
  h, grid_idx = pl.program_id(0), pl.program_id(1)
  j = sk.to_i32(original[0][grid_idx])
  should_write = original[4][grid_idx].astype(jnp.bool_)
  sender = lax.axis_index(axis_name)
  if cp_barrier:

    @pl.when(jnp.logical_and(h == 0, grid_idx == 0))
    def _cp_entry_barrier():
      # Only context coordinates change. Every peer consumes this
      # generation before sending data, and the final receive drain
      # prevents a later call from overtaking an unconsumed generation.
      barrier = pltpu.get_barrier_semaphore()

      def announce(offset, _):
        peer = lax.rem(sender + offset + 1, cp)
        pl.semaphore_signal(barrier, 1, device_id={axis_name: peer})

      lax.fori_loop(0, cp - 1, announce, None, unroll=False)
      pl.semaphore_wait(barrier, cp - 1)

  ordinal = h * kv_steps + j
  slot = lax.rem(ordinal, depth)

  # @reference.begin dkv_export_mapping
  # All logical output elements: source[c,h,j*4096+u,d] ->
  # destination[j//T,c*4+h,(j%T)*4096+u,d]. No arithmetic or reduction here.
  def copies(message):
    mh, mj = lax.div(message, kv_steps), lax.rem(message, kv_steps)
    ms = lax.rem(message, depth)
    peer, tile = lax.div(mj, tiles_per_peer), lax.rem(mj, tiles_per_peer)
    head = sender * heads + mh
    dst_k = out_k.at[head, pl.ds(tile * bkv, bkv), :]
    dst_v = out_v.at[head, pl.ds(tile * bkv, bkv), :]
    return (
        pltpu.make_async_remote_copy(
            stage_k.at[ms], dst_k, send_sem.at[ms, 0], recv_sem.at[sender, mh, tile, 0], device_id={axis_name: peer}
        ),
        pltpu.make_async_remote_copy(
            stage_v.at[ms], dst_v, send_sem.at[ms, 1], recv_sem.at[sender, mh, tile, 1], device_id={axis_name: peer}
        ),
    )

  # @reference.end dkv_export_mapping

  # @reference.begin dkv_stage_ownership
  # A slot is dead until its first write. Subsequent writes are protected by
  # the previous generation's send completion. Interior q accumulation does
  # not access these BF16 output slots, so it may proceed while DMA runs.
  if not serial:

    @pl.when(jnp.logical_and(should_write, ordinal >= depth))
    def _():
      for transfer in copies(ordinal - depth):
        transfer.wait_send()

  original[20], original[21] = stage_k.at[slot], stage_v.at[slot]
  sk._flash_attention_dkv_kernel(*original, **params)

  @pl.when(should_write)
  def _():
    transfers = copies(ordinal)
    for transfer in transfers:
      transfer.start()
    if serial:
      for transfer in transfers:
        transfer.wait_send()

  # @reference.end dkv_stage_ownership

  # @reference.begin dkv_exit_drain
  # Every sender emits heads*kv_steps tiles per tensor. Every receiver gets
  # CP*heads*T = heads*kv_steps distinct tiles; no output is consumed before
  # this full drain. A new call gets fresh semaphore lifetimes plus the
  # compiler's entry barrier. All semaphores must be zero on return.
  last = jnp.logical_and(h == heads - 1, grid_idx == pl.num_programs(1) - 1)

  @pl.when(last)
  def _():
    if not serial:
      for offset in range(depth):
        for transfer in copies(jnp.int32(heads * kv_steps - depth + offset)):
          transfer.wait_send()

    def receive_one(index, _):
      src = lax.div(index, heads * tiles_per_peer)
      rest = lax.rem(index, heads * tiles_per_peer)
      head, tile = lax.div(rest, tiles_per_peer), lax.rem(rest, tiles_per_peer)
      global_head = src * heads + head
      # These descriptors only wait; they never launch a local copy.
      # The local destination view gives exactly the remote send bytes.
      pltpu.make_async_copy(
          stage_k.at[0], out_k.at[global_head, pl.ds(tile * bkv, bkv), :], recv_sem.at[src, head, tile, 0]
      ).wait()
      pltpu.make_async_copy(
          stage_v.at[0], out_v.at[global_head, pl.ds(tile * bkv, bkv), :], recv_sem.at[src, head, tile, 1]
      ).wait()

    lax.fori_loop(0, cp * heads * tiles_per_peer, receive_one, None, unroll=False)


def output_kernel(*refs, cp, axis_name, cp_barrier, depth, serial, heads, kv_steps, dq_pipeline, **params):
  assert len(refs) == 39
  # Added output at index22 shifts original scratch and dKV scratch by one.
  original = list(refs[:22]) + list(refs[23:26])
  dkv_extra = list(refs[26:30])
  dq_final = refs[22]
  input_stage, output_stage, load_sem, store_sem, partials, partial_sem, export_stage, send, recv = refs[30:]
  dq_input, dq_output = original[16], original[19]
  h, g = pl.program_id(0), pl.program_id(1)
  n = pl.num_programs(1)
  bq, bkv = params["bq"], params["bkv"]
  assert bkv == 8 * bq
  slot = lax.rem(g, 2) if dq_pipeline else jnp.int32(0)
  j, i = sk.to_i32(original[0][g]), sk.to_i32(original[1][g])
  final_q = j == lax.div(i, 8)
  sender = lax.axis_index(axis_name)
  per_peer = kv_steps // cp

  def address(index):
    kj, qi = sk.to_i32(original[0][index]), sk.to_i32(original[1][index])
    return kj, qi

  def load(index):
    kj, qi = address(index)
    s = lax.rem(index, 2) if dq_pipeline else jnp.int32(0)
    return pltpu.make_async_copy(dq_input.at[lax.rem(kj, 3), h, pl.ds(qi * bq, bq), :], input_stage.at[s], load_sem.at[s])

  def store(index):
    kj, qi = address(index)
    s = lax.rem(index, 2) if dq_pipeline else jnp.int32(0)
    return pltpu.make_async_copy(
        output_stage.at[s], dq_output.at[lax.rem(kj, 3), h, pl.ds(qi * bq, bq), :], store_sem.at[s]
    )

  # @reference.begin dq_independent_input_output
  # Pinned production dq_alias -> dq_ref update: same BF16 arithmetic.
  # Input slot g%2 is read only; output slot g%2 is a distinct allocation.
  # Prefetch g+1 after the old input consumer, without waiting on output.
  # Wait store g-2 before overwriting its output bank. Metadata requires
  # partial-address revisit distance >2; no pending writer aliases lookahead.
  @pl.when(g == 0)
  def _():
    load(g).start()

  load(g).wait()

  @pl.when(g + 1 < n)
  def _():
    load(g + 1).start()

  @pl.when(g >= 2)
  def _():
    store(g - 2).wait()

  original[16] = input_stage.at[slot]
  original[19] = output_stage.at[slot]
  # @reference.end dq_independent_input_output
  dkv_output_kernel(
      *original,
      *dkv_extra,
      cp=cp,
      axis_name=axis_name,
      cp_barrier=cp_barrier,
      depth=depth,
      serial=serial,
      heads=heads,
      kv_steps=kv_steps,
      **params,
  )
  store(g).start()
  if not dq_pipeline:
    store(g).wait()
  # Diagnostic pair: retain final partial writeback in both controls.

  def transfer(message):
    mh, mj = lax.div(message, kv_steps), lax.rem(message, kv_steps)
    peer, tile = lax.div(mj, per_peer), lax.rem(mj, per_peer)
    return pltpu.make_async_remote_copy(
        export_stage.at[0],
        dq_final.at[sender * heads + mh, pl.ds(tile * bkv, bkv), :],
        send.at[0],
        recv.at[sender, mh, tile],
        device_id={axis_name: peer},
    )

  # @reference.begin dq_final_reduction_and_export
  # On the causal diagonal, all contributions to Q tile i are complete.
  # Other partials' last stores occurred >1 program ago, hence were waited.
  # Preserve BF16 partial updates, FP32 three-slot sum, then one BF16 cast.
  @pl.when(final_q)
  def _():
    message = h * kv_steps + j
    sub = lax.rem(i, 8)
    if dq_pipeline:

      @pl.when(jnp.logical_and(sub == 0, message > 0))
      def _():
        transfer(message - 1).wait_send()

    current = lax.rem(j, 3)
    partials[current] = output_stage[slot]
    for s in range(3):

      @pl.when(current != s)
      def _():
        pltpu.make_async_copy(dq_input.at[s, h, pl.ds(i * bq, bq), :], partials.at[s], partial_sem.at[s]).start()

    for s in range(3):

      @pl.when(current != s)
      def _():
        pltpu.make_async_copy(dq_input.at[s, h, pl.ds(i * bq, bq), :], partials.at[s], partial_sem.at[s]).wait()

    export_stage[0, pl.ds(sub * bq, bq), :] = jnp.sum(partials[...].astype(jnp.float32), axis=0).astype(
        export_stage.dtype
    )

    @pl.when(sub == 7)
    def _():
      transfer(message).start()
      if not dq_pipeline:
        transfer(message).wait_send()

  # @reference.end dq_final_reduction_and_export

  # @reference.begin dq_drain
  if dq_pipeline:

    @pl.when(g == n - 1)
    def _():
      @pl.when(g > 0)
      def _():
        store(g - 1).wait()

      store(g).wait()

  @pl.when(jnp.logical_and(h == heads - 1, g == n - 1))
  def _():
    if dq_pipeline:
      transfer(jnp.int32(heads * kv_steps - 1)).wait_send()

    def drain(index, _):
      src = lax.div(index, heads * per_peer)
      rest = lax.rem(index, heads * per_peer)
      head, tile = lax.div(rest, per_peer), lax.rem(rest, per_peer)
      pltpu.make_async_copy(
          export_stage.at[0], dq_final.at[src * heads + head, pl.ds(tile * bkv, bkv), :], recv.at[src, head, tile]
      ).wait()

    lax.fori_loop(0, cp * heads * per_peer, drain, None, unroll=False)


def import_output_kernel(*refs, base_kernel, n_pre, n_scratch, cp, axis_name, import_head, length):
  """Import the next head while the unchanged backward/export kernel runs.

  The compiler entry barrier makes fresh remote destinations valid (device
  probe P1a). Drain imports after base exports so all semaphores are empty
  before the next call consumes the fresh outputs.
  """
  assert n_pre == 19 and n_scratch == 16 and len(refs) == 49
  assert pl.num_programs(0) == 1
  sources = refs[n_pre : n_pre + 4]
  output_start = n_pre + 4
  outputs = refs[output_start + 4 : output_start + 8]
  scratch_start = output_start + 8
  original = refs[:n_pre] + refs[output_start : output_start + 4] + refs[scratch_start : scratch_start + n_scratch]
  send, recv = refs[-2:]
  assert len(original) == 39 and len(sources) == len(outputs) == 4
  sender = lax.axis_index(axis_name)

  def copies(peer):
    return tuple(
        pltpu.make_async_remote_copy(
            sources[t].at[peer * 4 + import_head],
            outputs[t].at[0, pl.ds(sender * length, length), :],
            send.at[t],
            recv.at[sender, t],
            device_id={axis_name: peer},
        )
        for t in range(4)
    )

  @pl.when(jnp.logical_and(pl.program_id(0) == 0, pl.program_id(1) == 0))
  def launch():
    def start_peer(peer, _):
      for transfer in copies(peer):
        transfer.start()

    lax.fori_loop(0, cp, start_peer, None, unroll=False)

  base_kernel(*original)

  # The original kernel drains its dQ/dKV exports before these imports.
  @pl.when(
      jnp.logical_and(
          pl.program_id(0) == pl.num_programs(0) - 1,
          pl.program_id(1) == pl.num_programs(1) - 1,
      )
  )
  def drain():
    def wait_peer(peer, _):
      for transfer in copies(peer):
        transfer.wait_send()

    lax.fori_loop(0, cp, wait_peer, None, unroll=False)

    def wait_source(source, _):
      for t in range(4):
        # Wait-only descriptor: source shape determines the received byte count.
        pltpu.make_async_copy(
            sources[t].at[import_head],
            outputs[t].at[0, pl.ds(source * length, length), :],
            recv.at[source, t],
        ).wait()

    lax.fori_loop(0, cp, wait_source, None, unroll=False)


def kv_shadow_output_kernel(*refs, base_kernel, n_pre, cp, axis_name, heads, length, prefetch):
  """Diagnostic K/V traffic alongside unchanged separate-input arithmetic.

  Each (source rank, head, destination rank, tensor) has unique source and
  receive counters. Fresh HBM outputs never feed arithmetic or reuse storage.
  The compiler global entry barrier establishes remote buffer availability.
  This intentionally duplicates input traffic; it is not a production path.
  """
  assert n_pre == 19 and len(refs) == n_pre + 26
  sources, outputs = refs[n_pre : n_pre + 2], refs[n_pre + 6 : n_pre + 8]
  original = refs[:n_pre] + refs[n_pre + 2 : n_pre + 6] + refs[n_pre + 8 : n_pre + 24]
  send, recv = refs[-2:]
  rank = lax.axis_index(axis_name)
  head, program = pl.program_id(0), pl.program_id(1)

  # @reference.begin kv_shadow_ownership
  # Diagnostic only: source[peer*heads+h, u, d] on rank c is copied to
  # output[h, c*length+u, d] on peer. Each complete target is written once.
  # There is no arithmetic consumer and no buffer/semaphore reuse.
  def copies(h, peer):
    return tuple(
        pltpu.make_async_remote_copy(
            sources[t].at[peer * heads + h],
            outputs[t].at[h, pl.ds(rank * length, length), :],
            send.at[h, peer, t],
            recv.at[h, rank, t],
            device_id={axis_name: peer},
        )
        for t in range(2)
    )

  def launch(h):
    def push(offset, _):
      peer = lax.rem(rank + offset, cp)
      for transfer in copies(h, peer):
        transfer.start()

    lax.fori_loop(0, cp, push, None, unroll=False)

  def drain(h):
    def wait_peer(peer, _):
      for transfer in copies(h, peer):
        transfer.wait_send()
      for t in range(2):
        pltpu.make_async_copy(
            sources[t].at[rank * heads + h],
            outputs[t].at[h, pl.ds(peer * length, length), :],
            recv.at[h, peer, t],
        ).wait()

    lax.fori_loop(0, cp, wait_peer, None, unroll=False)

  @pl.when(program == 0)
  def start_head():
    if prefetch:

      @pl.when(head == 0)
      def bootstrap():
        launch(jnp.int32(0))

      @pl.when(head + 1 < heads)
      def next_head():
        launch(head + 1)

    else:
      launch(head)
      drain(head)

  base_kernel(*original)

  if prefetch:

    @pl.when(jnp.logical_and(head == heads - 1, program == pl.num_programs(1) - 1))
    def drain_all():
      lax.fori_loop(0, heads, lambda h, _: drain(h), None, unroll=False)

  # @reference.end kv_shadow_ownership


def kv_previous_send(generation, *, rows, cp):
  """Previous generation using this producer's same-parity send semaphore.

  Each owner has a contiguous row interval in each head. At its first two
  rows the predecessor belongs to the previous head, not the previous owner.
  A negative result means this slot has not sent anything in this invocation.
  Even rows keep each owner's bank parity unchanged across head boundaries.
  """
  if cp <= 0 or rows < cp or rows % cp or rows % 2:
    raise ValueError("K/V send predecessors require even rows and complete owner partitions")
  per_owner = rows // cp
  offset = generation % per_owner
  previous_head_end = generation - offset - rows + per_owner - 1
  previous_head = previous_head_end - (previous_head_end - generation) % 2
  return jnp.where(offset >= 2, generation - 2, previous_head)


def kv_hbm_output_kernel(*refs, base_kernel, cp, axis_name, heads, rows, bq, prefetch, trace_regions=False):
  """Import into unique HBM head slots, then explicitly load two VMEM banks.

  Experimental only. Each sender writes its disjoint sequence chunk once per
  head and peer. An owner receive is consumed before its first local load.
  HBM slots and remote counters are never reused within a call. A full-dQ
  local copy fences the old bank's arithmetic before subsequent row loads.
  Hardware visibility and that last-read dependency need device acceptance.
  """
  assert len(refs) == 48 and heads == 4 and rows % cp == 0 and rows % 2 == 0
  original = list(refs[:23]) + list(refs[26:42])
  destinations, fence_output = refs[23:25], refs[25]
  k_bank, v_bank, send, recv, load_sem, fence_sem = refs[42:]
  sources = original[7:9]
  banks = (k_bank, v_bank)
  dq_stage = original[31]
  rank = lax.axis_index(axis_name)
  head, program = pl.program_id(0), pl.program_id(1)
  row = sk.to_i32(original[0][program])
  first = original[3][program].astype(jnp.bool_)
  last = original[4][program].astype(jnp.bool_)
  per_owner, block = rows // cp, 8 * bq
  length = per_owner * block
  bank = lax.rem(row, 2)

  def scope(name):
    return jax.named_scope(name) if trace_regions else contextlib.nullcontext()

  # @reference.begin kv_hbm_supply
  def transfers(h, peer):
    return tuple(
        pltpu.make_async_remote_copy(
            sources[t].at[peer * heads + h],
            destinations[t].at[h, pl.ds(rank * length, length), :],
            send.at[h, peer, t],
            recv.at[h, rank, t],
            device_id={axis_name: peer},
        )
        for t in range(2)
    )

  def launch(h):
    def push(offset, _):
      peer = lax.rem(rank + offset, cp)
      for transfer in transfers(h, peer):
        transfer.start()

    lax.fori_loop(0, cp, push, None, unroll=False)

  def receive(h, r, region):
    @pl.when(lax.rem(r, per_owner) == 0)
    def owner_first_use():
      owner = lax.div(r, per_owner)
      with scope(region):
        for t in range(2):
          # Wait-only descriptor: same complete owner chunk as the remote DMA.
          pltpu.make_async_copy(
              sources[t].at[rank * heads + h],
              destinations[t].at[h, pl.ds(owner * length, length), :],
              recv.at[h, owner, t],
          ).wait()

  def loads(h, r):
    slot = lax.rem(r, 2)
    return tuple(
        pltpu.make_async_copy(
            destinations[t].at[h, pl.ds(r * block, block), :],
            banks[t].at[slot],
            load_sem.at[slot, t],
        )
        for t in range(2)
    )

  @pl.when(program == 0)
  def start_head():
    if prefetch:

      @pl.when(head == 0)
      def bootstrap():
        launch(jnp.int32(0))

    else:
      launch(head)
    receive(head, jnp.int32(0), "mla_hbm_head_recv")
    for transfer in loads(head, jnp.int32(0)):
      transfer.start()
    # Supply the next head only after this head's first owner has arrived.
    # Enqueue the current local load before the next head's remote traffic.
    # No remote bank credits or per-head all-peer send drains are required.
    if prefetch:

      @pl.when(head + 1 < heads)
      def next_head():
        launch(head + 1)

  # @reference.end kv_hbm_supply
  # @reference.begin kv_hbm_consumer_banks
  @pl.when(first)
  def use_row():
    if not prefetch:

      @pl.when(row > 0)
      def serial_load():
        receive(head, row, "mla_hbm_owner_recv")
        for transfer in loads(head, row):
          transfer.start()

    with scope("mla_hbm_load_wait"):
      for transfer in loads(head, row):
        transfer.wait()
    if prefetch:

      @pl.when(row + 1 < rows)
      def next_row():
        receive(head, row + 1, "mla_hbm_owner_recv")
        for transfer in loads(head, row + 1):
          transfer.start()

  original[7], original[8] = k_bank.at[bank], v_bank.at[bank]
  base_kernel(*original)

  @pl.when(last)
  def release_row():
    # Keep the entire last Q-program's dQ payload as the local dependency.
    # Waiting on the local DMA also protects this stage and fence destination
    # before their next reuse. Do not replace it with a scalar signal.
    with scope("mla_hbm_row_fence"):
      fence = pltpu.make_async_copy(dq_stage.at[lax.rem(program, 2)], fence_output, fence_sem.at[0])
      fence.start()
      fence.wait()

  # @reference.end kv_hbm_consumer_banks
  # Base gradient exports have already drained before the final invocation
  # returns. Every HBM receive was consumed at first use, including self.
  @pl.when(jnp.logical_and(head == heads - 1, program == pl.num_programs(1) - 1))
  def drain_sources():
    def wait_head(h, _):
      def wait_peer(peer, _):
        for transfer in transfers(h, peer):
          transfer.wait_send()

      lax.fori_loop(0, cp, wait_peer, None, unroll=False)

    with scope("mla_hbm_send_drain"):
      lax.fori_loop(0, heads, wait_head, None, unroll=False)


def kv_import_output_kernel(
    *refs, base_kernel, cp, axis_name, heads, rows, bq, prefetch, defer_send=False, launch_after_credit=False
):
  """Import each K/V row once, with a full-dQ DMA as its consumer credit.

  This experimental wrapper preserves the pinned arithmetic. Its credit DMA
  reads the entire last Q-program's dQ output, so releasing a bank depends on
  the dQ path's K/V reads as well as dK/dV. The mailbox payload is discarded;
  receive completion grants ownership. No scalar credit-signal fence is used.
  """
  assert len(refs) == 46
  # One extra HBM output precedes the original sixteen scratch operands.
  original = list(refs[:23]) + list(refs[24:40])
  mailbox = refs[23]
  k_bank, v_bank, send, recv, credit_send, credit_recv = refs[40:]
  k_source, v_source = original[7:9]
  dq_stage = original[31]
  rank = lax.axis_index(axis_name)
  head, program = pl.program_id(0), pl.program_id(1)
  row_index = sk.to_i32(original[0][program])
  generation = head * rows + row_index
  bank = lax.rem(generation, 2)
  first = original[3][program].astype(jnp.bool_)
  last = original[4][program].astype(jnp.bool_)
  generations, per_owner, block = heads * rows, rows // cp, 8 * bq

  def owner(g):
    return lax.div(lax.rem(g, rows), per_owner)

  def source(ref, g, peer):
    return ref.at[peer * heads + lax.div(g, rows), pl.ds(lax.rem(g, per_owner) * block, block), :]

  def remote(g, peer, tensor):
    src, dst = (k_source, k_bank) if tensor == 0 else (v_source, v_bank)
    slot = lax.rem(g, 2)
    return pltpu.make_async_remote_copy(
        source(src, g, peer),
        dst.at[slot],
        send.at[slot, peer, tensor],
        recv.at[slot, tensor],
        device_id={axis_name: peer},
    )

  def launch(g):
    @pl.when(rank == owner(g))
    def producer():
      slot = lax.rem(g, 2)
      if defer_send:
        previous = kv_previous_send(g, rows=rows, cp=cp)

        @pl.when(previous >= 0)
        def reuse_send_slot():
          wait_send(previous)

      @pl.when(g >= 2)
      def acquire():
        def wait_peer(peer, _):
          # Wait-only descriptor, sized exactly like the full dQ credit.
          pltpu.make_async_copy(dq_stage.at[0], mailbox.at[slot, peer], credit_recv.at[slot, peer]).wait()

        lax.fori_loop(0, cp, wait_peer, None, unroll=False)

      def push(peer, _):
        @pl.when(peer == rank)
        def loopback():
          pltpu.make_async_copy(source(k_source, g, peer), k_bank.at[slot], recv.at[slot, 0]).start()
          pltpu.make_async_copy(source(v_source, g, peer), v_bank.at[slot], recv.at[slot, 1]).start()

        @pl.when(peer != rank)
        def peer_copy():
          remote(g, peer, 0).start()
          remote(g, peer, 1).start()

      lax.fori_loop(0, cp, push, None, unroll=False)

  def wait_send(g):
    @pl.when(rank == owner(g))
    def producer():
      def peer_wait(peer, _):
        @pl.when(peer != rank)
        def wait_remote():
          remote(g, peer, 0).wait_send()
          remote(g, peer, 1).wait_send()

      lax.fori_loop(0, cp, peer_wait, None, unroll=False)

  @pl.when(jnp.logical_and(head == 0, program == 0))
  def bootstrap():
    launch(jnp.int32(0))
    if not defer_send:
      wait_send(jnp.int32(0))
    if launch_after_credit:
      launch(jnp.int32(1))

  @pl.when(first)
  def receive_row():
    pltpu.make_async_copy(k_bank.at[bank], k_bank.at[bank], recv.at[bank, 0]).wait()
    pltpu.make_async_copy(v_bank.at[bank], v_bank.at[bank], recv.at[bank, 1]).wait()
    if prefetch and not launch_after_credit:

      @pl.when(generation + 1 < generations)
      def next_row():
        launch(generation + 1)

  original[7], original[8] = k_bank.at[bank], v_bank.at[bank]
  base_kernel(*original)

  @pl.when(last)
  def release_row():
    if prefetch and not defer_send:

      @pl.when(generation + 1 < generations)
      def drain_source():
        wait_send(generation + 1)

    @pl.when(generation + 2 < generations)
    def credit():
      # output_stage is indexed by Q-program parity, not KV-row parity.
      # Addressed remote copy also handles self, as in gradient export. Its
      # source wait protects this stage before the next program may reuse it.
      copy = pltpu.make_async_remote_copy(
          dq_stage.at[lax.rem(program, 2)],
          mailbox.at[bank, rank],
          credit_send.at[0],
          credit_recv.at[bank, rank],
          device_id={axis_name: owner(generation + 2)},
      )
      copy.start()
      copy.wait_send()
      if launch_after_credit:
        # All ranks release their local last read before the next producer
        # waits for every credit. Enqueue need not wait for receive(g + 1).
        launch(generation + 2)

    if not prefetch:

      @pl.when(generation + 1 < generations)
      def serial_next():
        launch(generation + 1)
        wait_send(generation + 1)

    if defer_send:

      @pl.when(generation == generations - 1)
      def drain_final_sends():
        # The final head contains the last use of every producer's live slot.
        # One-row owners only use one parity; do not wait on their other slot.
        start = (heads - 1) * rows + rank * per_owner
        end = start + per_owner - 1
        for slot in range(2):
          final = end - (end - slot) % 2

          @pl.when(final >= start)
          def drain():
            wait_send(final)


def fused_backward(
    q,
    k,
    v,
    segment_ids,
    logsumexp,
    do,
    di,
    *,
    cp: int,
    axis_name: str,
    cp_barrier: bool = False,
    import_head: int | None = None,
    imports: tuple | None = None,
    kv_import_prefetch: bool | None = None,
    kv_import_defer_send: bool = False,
    kv_import_launch_after_credit: bool = False,
    kv_hbm_prefetch: bool | None = None,
    kv_hbm_trace: bool = False,
    shadow_imports: tuple | None = None,
    shadow_prefetch: bool = False,
    bq: int,
    bkv: int,
    bkv_compute: int,
    is_mqa: bool,
    mask_info: MaskInfo,
    mask_value: float,
    mask_function: MaskFunctionType | None,
    config: SplashConfig,
    dkv_mask_sparsity: float,
):
  kv_hbm = kv_hbm_prefetch is not None
  if type(kv_hbm_trace) is not bool or (kv_hbm_trace and kv_hbm_prefetch is not True):
    raise ValueError("HBM region tracing requires prefetched HBM K/V imports")
  if kv_hbm and (type(kv_hbm_prefetch) is not bool or kv_import_prefetch is not None):
    raise ValueError("HBM K/V staging requires a static selector and no row-credit import")
  if kv_hbm and (k.dtype != jnp.bfloat16 or v.dtype != jnp.bfloat16):
    raise ValueError("HBM K/V staging requires BF16 sources")
  kv_import = kv_import_prefetch is not None or kv_hbm
  if kv_import_defer_send and kv_import_prefetch is not True:
    raise ValueError("Deferred K/V send waits require row prefetch")
  if kv_import_launch_after_credit and not kv_import_defer_send:
    raise ValueError("Launching after credit requires deferred K/V send waits")
  native_layout = (
      not kv_import
      and _use_native_layout(config, q, k, v, segment_ids, mask_info, mask_function, is_mqa=is_mqa)
      and bkv > bkv_compute
      and do.dtype == q.dtype
  )
  num_q_heads, q_seq_len, head_dim_qk = q.shape
  if shadow_imports is not None:
    if kv_import or import_head is not None or imports is not None or cp_barrier or num_q_heads != 4:
      raise ValueError("Shadow K/V traffic requires separate four-head inputs and the global barrier")
    if len(shadow_imports) != 2 or type(shadow_prefetch) is not bool:
      raise ValueError("Shadow K/V traffic requires two sources and a static prefetch selector")
    for source, width in zip(shadow_imports, (192, 128), strict=True):
      if source.shape != (cp * 4, q_seq_len // cp, width) or source.dtype != jnp.bfloat16:
        raise ValueError("Shadow K/V sources require BF16 [CP*4,S/CP,D]")
  elif shadow_prefetch:
    raise ValueError("Shadow prefetch requires K/V sources")
  if kv_import:
    if (
        (not kv_hbm and type(kv_import_prefetch) is not bool)
        or num_q_heads != 4
        or cp_barrier
        or import_head is not None
        or imports is not None
    ):
      raise ValueError("K/V row import requires four heads, global barrier and no per-head import")
    if k.shape != (cp * 4, q_seq_len // cp, 192) or v.shape != (cp * 4, q_seq_len // cp, 128):
      raise ValueError("K/V row import requires sequence-owned [CP*4,S/CP,D] sources")
  if import_head is None:
    if imports is not None:
      raise ValueError("Import sources require import_head")
  else:
    if num_q_heads != 1 or cp_barrier or type(import_head) is not int or import_head not in (1, 2, 3):
      raise ValueError("Next-head import requires one head, global barrier, and import_head in (1,2,3)")
    if q_seq_len % cp or imports is None or len(imports) != 4:
      raise ValueError("Next-head import requires four sequence-sharded sources")
    for source, width in zip(imports, (192, 192, 128, 128), strict=True):
      if source.shape != (cp * 4, q_seq_len // cp, width) or source.dtype != jnp.bfloat16:
        raise ValueError("Next-head import sources must be BF16 [CP*4,S/CP,D]")
  kv_seq_len, head_dim_v = v.shape[-2:]
  num_kv_heads = 1 if is_mqa else k.shape[0]
  if kv_import:
    kv_seq_len, num_kv_heads = q_seq_len, num_q_heads
  dynamic_grid = mask_info.active_rows is not None

  bounds_start, bounds_end = mask_info_lib.find_bounds(mask_info.active_rows)
  if bq > q_seq_len:
    raise ValueError(f"{bq=} should not be greater than {q_seq_len=}")
  if bkv > kv_seq_len:
    raise ValueError(f"{bkv=} should not be greater than {kv_seq_len=}")
  if bkv_compute > bkv:
    raise ValueError(f"{bkv_compute=} should not be greater than {bkv=}")
  if bkv % bkv_compute:
    raise ValueError(f"{bkv=} should be a multiple of {bkv_compute=}")

  if not is_mqa and num_q_heads % num_kv_heads != 0:
    raise ValueError(
        f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
        f" multiple of the number of 'query' heads ({num_q_heads})"
    )

  if k.shape[:-1] != v.shape[:-1]:
    raise ValueError(f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same " "leading dimensions.")

  kv_steps = kv_seq_len // bkv
  q_steps = q_seq_len // bq
  q_heads_per_kv_head = num_q_heads // num_kv_heads
  if dynamic_grid:

    def unravel(f):
      def index_map(h, grid_idx, rows_ref, cols_ref, *_):
        j = to_i32(rows_ref[grid_idx])
        i = to_i32(cols_ref[grid_idx])
        return f(h, i, j)

      return index_map

    grid_size = mask_info.num_active_blocks[0]
    grid = (num_q_heads, grid_size)

    def mask_index_map(h, grid_idx, rows_ref, cols_ref, mask_next_ref=None, *_):
      del h, rows_ref, cols_ref  # Unused.
      next_m = to_i32(mask_next_ref[grid_idx])
      return next_m, 0, 0

  else:
    unravel = lambda f: lambda j, h, i, *_: f(h, i, j)
    grid = (kv_steps, num_q_heads, q_steps)

    def mask_index_map(j, h, i, rows_ref, cols_ref, mask_next_ref=None, *_):
      del h, rows_ref, cols_ref  # Unused.
      grid_idx = j * q_steps + i
      next_m = to_i32(mask_next_ref[grid_idx])
      return next_m, 0, 0

  q_index_map = unravel(lambda h, i, j: from_head_minor((h, i, 0), config.q_layout))
  o_index_map = unravel(lambda h, i, j: (h, i, 0))

  def create_kv_index_map(layout):
    def index_map(h, i, j, *_):
      del i  # Unused.
      prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
      return from_head_minor((*prefix, j, 0), layout)

    return index_map

  k_index_map = unravel(create_kv_index_map(config.k_layout))
  v_index_map = unravel(create_kv_index_map(config.v_layout))

  q_spec = pl.BlockSpec(from_head_minor((None, bq, head_dim_qk), config.q_layout), q_index_map)

  o_spec = pl.BlockSpec((None, bq, head_dim_v), o_index_map)
  k_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
          config.k_layout,
      ),
      k_index_map,
  )

  v_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v),
          config.v_layout,
      ),
      v_index_map,
  )
  if kv_import:
    k_spec = v_spec = pl.BlockSpec(memory_space=pl.ANY)

  def create_dkv_index_map(h, i, j, *_):
    del i  # Unused.
    prefix = () if is_mqa else (_div(h, q_heads_per_kv_head),)
    layout = QKVLayout.SEQ_MINOR if native_layout else QKVLayout.HEAD_DIM_MINOR
    return from_head_minor((*prefix, j, 0), layout)

  dkv_index_map = unravel(create_dkv_index_map)

  dk_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
          QKVLayout.SEQ_MINOR if native_layout else QKVLayout.HEAD_DIM_MINOR,
      ),
      dkv_index_map,
  )

  dv_spec = pl.BlockSpec(
      from_head_minor(
          (bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v),
          QKVLayout.SEQ_MINOR if native_layout else QKVLayout.HEAD_DIM_MINOR,
      ),
      dkv_index_map,
  )
  mask_spec = pl.BlockSpec((None, bkv, bq), mask_index_map)

  q_segment_ids_index_map = unravel(lambda h, i, j: (0, i))
  if segment_ids is not None and segment_ids.q.ndim == 2:
    q_segment_spec, q_segment_ids = _group_segment_spec(
        segment_ids.q,
        num_q_heads,
        bq,
        True,
        1 if native_layout else NUM_SUBLANES,
        0,
        unravel,
    )
    kv_segment_spec, kv_segment_ids = _group_segment_spec(
        segment_ids.kv,
        num_q_heads,
        bkv,
        native_layout,
        1 if native_layout else NUM_LANES,
        1,
        unravel,
    )
  elif segment_ids is not None:
    kv_segment_ids_index_map = unravel(lambda h, i, j: (j, 0))
    if native_layout:
      # MaskInfo already fixes the DMA tiles; only physical buffer layouts
      # change here, not the visited blocks or segment equality semantics.
      q_segment_spec = pl.BlockSpec((1, bq), q_segment_ids_index_map)
      kv_segment_spec = pl.BlockSpec((1, bkv), unravel(lambda h, i, j: (0, j)))
      q_segment_ids = segment_ids.q[None, :]
      kv_segment_ids = segment_ids.kv[None, :]
    else:
      q_segment_spec = pl.BlockSpec((NUM_SUBLANES, bq), q_segment_ids_index_map)
      kv_segment_spec = pl.BlockSpec((bkv, NUM_LANES), kv_segment_ids_index_map)
      q_segment_ids = jax.lax.broadcast_in_dim(segment_ids.q, (NUM_SUBLANES, q_seq_len), (1,))
      kv_segment_ids = jax.lax.broadcast_in_dim(segment_ids.kv, (kv_seq_len, NUM_LANES), (0,))
  else:
    q_segment_spec = kv_segment_spec = None
    q_segment_ids = kv_segment_ids = None

  if native_layout:
    do_spec = pl.BlockSpec((None, head_dim_v, bq), unravel(lambda h, i, j: (h, 0, i)))
  else:
    do_spec = o_spec

  logsumexp_index_map = unravel(lambda h, i, j: (h, 0, i))

  assert logsumexp.shape == di.shape == (num_q_heads, q_seq_len)
  # TODO: Remove the sublane expansion once Mosaic has all retilings
  logsumexp_shape = (num_q_heads, NUM_SUBLANES, q_seq_len)
  logsumexp = jnp.broadcast_to(jnp.expand_dims(logsumexp, -2), logsumexp_shape)
  logsumexp_spec = pl.BlockSpec((None, NUM_SUBLANES, bq), logsumexp_index_map)
  assert logsumexp.ndim == len(logsumexp_spec.block_shape)

  # TODO: Remove the sublane expansion once Mosaic has all retilings
  di = jnp.broadcast_to(jnp.expand_dims(di, -2), logsumexp_shape)
  di_spec = pl.BlockSpec((None, NUM_SUBLANES, bq), logsumexp_index_map)
  assert di.ndim == len(di_spec.block_shape)

  in_specs = [
      q_spec,
      k_spec,
      v_spec,
      q_segment_spec,
      kv_segment_spec,
      logsumexp_spec,
      do_spec,
      di_spec,
  ]
  if mask_info.partial_mask_blocks is not None:
    in_specs.append(mask_spec)
  else:
    in_specs.append(None)

  if mask_info.q_sequence is not None:
    in_specs.append(pl.BlockSpec((NUM_SUBLANES, bq), q_segment_ids_index_map))
    q_sequence = jax.lax.broadcast_in_dim(mask_info.q_sequence, (NUM_SUBLANES, q_seq_len), (1,))
  else:
    q_sequence = None
    in_specs.append(None)

  dq_reduction_steps = config.dq_reduction_steps
  if not dynamic_grid and kv_steps <= 3 and dq_reduction_steps == 3:
    dq_reduction_steps = None

  dq = dq_alias_spec = None
  dq_layout = QKVLayout.SEQ_MINOR if native_layout else QKVLayout.HEAD_DIM_MINOR
  dq_block_shape = (None, None, *from_head_minor((bq, head_dim_qk), dq_layout))
  dq_output_shape = from_head_minor(q.shape, dq_layout)
  if dq_reduction_steps == 3:
    dq_index_map = unravel(lambda h, i, j: from_head_minor((j % 3, h, i, 0), dq_layout))
    dq_spec = pl.BlockSpec(dq_block_shape, dq_index_map)
    dq_alias_spec = dq_spec
    dq_shape = jax.ShapeDtypeStruct((3, *dq_output_shape), q.dtype)
    dq = jnp.zeros_like(dq_shape)
  else:
    dq_index_map = unravel(lambda h, i, j: from_head_minor((j, h, i, 0), dq_layout))
    dq_spec = pl.BlockSpec(dq_block_shape, dq_index_map)
    # Only accumulate in fp32 if there's a small number of reduction steps.
    q_dtype = q.dtype if kv_steps <= 4 else jnp.float32
    dq_shape = jax.ShapeDtypeStruct((kv_steps, *dq_output_shape), q_dtype)

  assert dq_reduction_steps == 3 and dynamic_grid and not native_layout
  dq_spec = dq_alias_spec = pl.BlockSpec(memory_space=pl.ANY)
  in_specs += [dq_alias_spec]

  if bkv == bkv_compute:
    dq_scratch = None
  else:
    dq_scratch = pltpu.VMEM(
        ((head_dim_qk, bq) if native_layout else (bq, head_dim_qk)),
        jnp.float32,
    )

  if dynamic_grid and q_heads_per_kv_head != 1:
    # in/out aliasing to accumulate within kv groups.
    in_specs += [dk_spec, dv_spec]
    dk = lax.empty(k.shape, dtype=jnp.float32)
    dv = lax.empty(v.shape, dtype=jnp.float32)
    # Keep gradients in fp32 when accumulating over head groups.
    dk_type = dv_type = jnp.float32
  else:
    in_specs += [None, None]
    dk, dv = None, None
    dk_type = k.dtype
    dv_type = v.dtype

  dk_shape = k.mT.shape if native_layout else k.shape
  dv_shape = v.mT.shape if native_layout else v.shape
  out_shapes = [
      dq_shape,
      jax.ShapeDtypeStruct(dk_shape, dk_type),
      jax.ShapeDtypeStruct(dv_shape, dv_type),
  ]
  assert q_heads_per_kv_head == 1 and num_q_heads in (1, 4)
  out_shapes[1] = jax.ShapeDtypeStruct((num_q_heads * cp, kv_seq_len // cp, head_dim_qk), dk_type)
  out_shapes[2] = jax.ShapeDtypeStruct((num_q_heads * cp, kv_seq_len // cp, head_dim_v), dv_type)
  out_shapes.append(jax.ShapeDtypeStruct((num_q_heads * cp, q_seq_len // cp, head_dim_qk), q.dtype))
  out_specs = [pl.BlockSpec(memory_space=pl.ANY) for _ in range(4)]

  kernel = functools.partial(
      output_kernel,
      cp=cp,
      axis_name=axis_name,
      cp_barrier=cp_barrier,
      depth=1,
      serial=False,
      heads=num_q_heads,
      kv_steps=kv_steps,
      dq_pipeline=True,
      mask_value=mask_value,
      q_steps=q_steps,
      bq=bq,
      bkv_compute=bkv_compute,
      config=config,
      bkv=bkv,
      mask_function=mask_function,
      q_heads_per_kv_head=q_heads_per_kv_head,
      native_layout=native_layout,
  )

  kernel_name = get_kernel_name(
      is_mqa=is_mqa,
      save_residuals=False,
      is_segmented=segment_ids is not None,
      phase="dkv",
  )
  metadata = {
      "xprof_metadata": json.dumps(
          dict(
              block_q_dkv=bq,
              block_kv_dkv=bkv,
              block_kv_dkv_compute=bkv_compute,
              q_layout=config.q_layout,
              k_layout=config.k_layout,
              v_layout=config.v_layout,
              use_experimental_scheduler=config.use_experimental_scheduler,
          ),
      )
  }
  args = [
      # scalar prefetch
      mask_info.active_rows,
      mask_info.active_cols,
      mask_info.mask_next,
      bounds_start,
      bounds_end,
      mask_info.block_mask,
      # inputs
      q if config.q_layout == QKVLayout.HEAD_DIM_MINOR else q.mT,
      k if config.k_layout == QKVLayout.HEAD_DIM_MINOR else k.mT,
      v if config.v_layout == QKVLayout.HEAD_DIM_MINOR else v.mT,
      q_segment_ids,
      kv_segment_ids,
      logsumexp,
      do.mT if native_layout else do,
      di,
      mask_info.partial_mask_blocks,
      q_sequence,
  ]
  input_fusion_candidates = [
      False,  # active_rows
      False,  # active_cols
      False,  # mask_next
      False,  # bounds_start
      False,  # bounds_end
      False,  # block_mask
      False,  # q
      False,  # k
      False,  # v
      # Group-indexed ID loads cannot always be dematerialized by input fusion.
      segment_ids is None or segment_ids.q.ndim == 1,  # q_segment_ids
      segment_ids is None or segment_ids.kv.ndim == 1,  # kv_segment_ids
      False,  # logsumexp
      False,  # do
      False,  # di
      False,  # partial_mask_blocks
      False,  # q_sequence
  ]
  num_args = sum(1 for x in args if x is not None)
  input_output_aliases = {}
  if dq_reduction_steps == 3:
    if dynamic_grid and q_heads_per_kv_head != 1:
      input_output_aliases = {num_args: 0, num_args + 1: 1, num_args + 2: 2}
    else:
      input_output_aliases = {num_args: 0}
  elif dynamic_grid and q_heads_per_kv_head != 1:
    input_output_aliases = {num_args: 1, num_args + 1: 2}

  scratch_shapes = [
      dq_scratch,
      pltpu.VMEM(
          ((head_dim_qk, bkv) if native_layout else (bkv, head_dim_qk)),
          jnp.float32,
      ),
      pltpu.VMEM(
          ((head_dim_v, bkv) if native_layout else (bkv, head_dim_v)),
          jnp.float32,
      ),
  ]

  scratch_shapes += [
      pltpu.VMEM((1, bkv, head_dim_qk), dk_type),
      pltpu.VMEM((1, bkv, head_dim_v), dv_type),
      pltpu.SemaphoreType.DMA((1, 2)),
      pltpu.SemaphoreType.DMA((cp, num_q_heads, kv_steps // cp, 2)),
      pltpu.VMEM((2, bq, head_dim_qk), q.dtype),
      pltpu.VMEM((2, bq, head_dim_qk), q.dtype),
      pltpu.SemaphoreType.DMA((2,)),
      pltpu.SemaphoreType.DMA((2,)),
      pltpu.VMEM((3, bq, head_dim_qk), q.dtype),
      pltpu.SemaphoreType.DMA((3,)),
      pltpu.VMEM((1, bkv, head_dim_qk), q.dtype),
      pltpu.SemaphoreType.DMA((1,)),
      pltpu.SemaphoreType.DMA((cp, num_q_heads, kv_steps // cp)),
  ]

  def _bwd_cost_estimate(
      q: jax.Array,
      k: jax.Array,
      v: jax.Array,
      q_segment_ids: jax.Array | None,
      kv_segment_ids: jax.Array | None,
      logsumexp: jax.Array,
      do: jax.Array,
      di: jax.Array,
      partial_mask_blocks: jax.Array | None,
      q_sequence: jax.Array | None,
      out_shapes: list[jax.ShapeDtypeStruct],
      mask_sparsity_factor: float,
  ) -> pl.CostEstimate:
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    kv_seq_len, head_dim_v = v.shape[-2:]
    if kv_import:
      kv_seq_len = q_seq_len

    total_matmul_flops_per_head = (
        2 * q_seq_len * kv_seq_len * head_dim_qk  # qk
        + 2 * q_seq_len * kv_seq_len * head_dim_v  # dv
        + 2 * q_seq_len * kv_seq_len * head_dim_v  # dp
        + 2 * q_seq_len * kv_seq_len * head_dim_qk  # dq
        + 2 * q_seq_len * kv_seq_len * head_dim_qk  # dk
    )

    estimated_flops = int(total_matmul_flops_per_head * num_q_heads * mask_sparsity_factor)

    exp_flops = num_q_heads * q_seq_len * kv_seq_len * mask_sparsity_factor
    if config.attn_logits_soft_cap is None:
      tanh_flops = 0
    else:
      tanh_flops = 2 * num_q_heads * q_seq_len * kv_seq_len * mask_sparsity_factor
    estimated_transcendentals = int(exp_flops + tanh_flops)

    inputs_ = [
        q,
        k,
        v,
        q_segment_ids,
        kv_segment_ids,
        logsumexp,
        do,
        di,
        partial_mask_blocks,
        q_sequence,
    ]
    input_bytes = sum(map(_bytes, inputs_))
    output_bytes = sum(map(_bytes, out_shapes))

    estimated_bytes = input_bytes + output_bytes

    return pl.CostEstimate(
        flops=estimated_flops,
        transcendentals=estimated_transcendentals,
        bytes_accessed=estimated_bytes,
    )

  cost_estimate = config.bwd_cost_estimate or _bwd_cost_estimate(
      q,
      k,
      v,
      q_segment_ids,
      kv_segment_ids,
      logsumexp,
      do,
      di,
      mask_info.partial_mask_blocks,
      q_sequence,
      out_shapes,
      dkv_mask_sparsity,
  )

  allow_input_fusion = None
  if native_layout:
    # The dynamic grid bound is the first custom-call operand. Pallas then
    # removes None arguments while preserving the order of the remaining
    # scalar-prefetch and regular inputs, followed by aliased outputs.
    allow_input_fusion = (
        *((False,) if dynamic_grid else ()),
        *(can_fuse for arg, can_fuse in zip(args, input_fusion_candidates) if arg is not None),
        *(False for value in (dq, dk, dv) if value is not None),
    )

  import_args = ()
  if kv_hbm:
    assert num_args == 14 and input_output_aliases == {14: 0} and len(scratch_shapes) == 16
    assert dynamic_grid and not native_layout and 6 + len(in_specs) == 19
    kernel = functools.partial(
        kv_hbm_output_kernel,
        base_kernel=kernel,
        cp=cp,
        axis_name=axis_name,
        heads=num_q_heads,
        rows=kv_steps,
        bq=bq,
        prefetch=kv_hbm_prefetch,
        trace_regions=kv_hbm_trace,
    )
    out_shapes += [jax.ShapeDtypeStruct((num_q_heads, q_seq_len, d), jnp.bfloat16) for d in (192, 128)]
    out_shapes.append(jax.ShapeDtypeStruct((bq, head_dim_qk), q.dtype))
    out_specs += [pl.BlockSpec(memory_space=pl.ANY) for _ in range(3)]
    scratch_shapes += [
        pltpu.VMEM((2, bkv, head_dim_qk), k.dtype),
        pltpu.VMEM((2, bkv, head_dim_v), v.dtype),
        pltpu.SemaphoreType.DMA((num_q_heads, cp, 2)),
        pltpu.SemaphoreType.DMA((num_q_heads, cp, 2)),
        pltpu.SemaphoreType.DMA((2, 2)),
        pltpu.SemaphoreType.DMA((1,)),
    ]
  elif kv_import:
    assert num_args == 14 and input_output_aliases == {14: 0} and len(scratch_shapes) == 16
    kernel = functools.partial(
        kv_import_output_kernel,
        base_kernel=kernel,
        cp=cp,
        axis_name=axis_name,
        heads=num_q_heads,
        rows=kv_steps,
        bq=bq,
        prefetch=kv_import_prefetch,
        defer_send=kv_import_defer_send,
        launch_after_credit=kv_import_launch_after_credit,
    )
    out_shapes.append(jax.ShapeDtypeStruct((2, cp, bq, head_dim_qk), q.dtype))
    out_specs.append(pl.BlockSpec(memory_space=pl.ANY))
    scratch_shapes += [
        pltpu.VMEM((2, bkv, head_dim_qk), k.dtype),
        pltpu.VMEM((2, bkv, head_dim_v), v.dtype),
        pltpu.SemaphoreType.DMA((2, cp, 2)),
        pltpu.SemaphoreType.DMA((2, 2)),
        pltpu.SemaphoreType.DMA((1,)),
        pltpu.SemaphoreType.DMA((2, cp)),
    ]
  if import_head is not None:
    # Append sources after the dQ alias, preserving flattened alias index 14.
    assert num_args == 14 and input_output_aliases == {14: 0}
    n_pre, n_scratch = 6 + len(in_specs), len(scratch_shapes)
    kernel = functools.partial(
        import_output_kernel,
        base_kernel=kernel,
        n_pre=n_pre,
        n_scratch=n_scratch,
        cp=cp,
        axis_name=axis_name,
        import_head=import_head,
        length=q_seq_len // cp,
    )
    import_args = tuple(imports)
    in_specs += [pl.BlockSpec(memory_space=pl.ANY) for _ in range(4)]
    out_shapes += [jax.ShapeDtypeStruct((1, q_seq_len, d), jnp.bfloat16) for d in (192, 192, 128, 128)]
    out_specs += [pl.BlockSpec(memory_space=pl.ANY) for _ in range(4)]
    scratch_shapes += [pltpu.SemaphoreType.DMA((4,)), pltpu.SemaphoreType.DMA((cp, 4))]

  if shadow_imports is not None:
    assert num_args == 14 and len(scratch_shapes) == 16 and not native_layout
    assert dynamic_grid and input_output_aliases == {14: 0}
    n_pre = 6 + len(in_specs)
    assert n_pre == 19
    kernel = functools.partial(
        kv_shadow_output_kernel,
        base_kernel=kernel,
        n_pre=n_pre,
        cp=cp,
        axis_name=axis_name,
        heads=num_q_heads,
        length=q_seq_len // cp,
        prefetch=shadow_prefetch,
    )
    import_args = tuple(shadow_imports)
    in_specs += [pl.BlockSpec(memory_space=pl.ANY) for _ in range(2)]
    out_shapes += [jax.ShapeDtypeStruct((num_q_heads, q_seq_len, d), jnp.bfloat16) for d in (192, 128)]
    out_specs += [pl.BlockSpec(memory_space=pl.ANY) for _ in range(2)]
    scratch_shapes += [pltpu.SemaphoreType.DMA((num_q_heads, cp, 2)) for _ in range(2)]

  with jax.named_scope(kernel_name):
    results = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shapes,
        input_output_aliases=input_output_aliases,
        # We set all dimensions to arbitrary because:
        # 1) for heads, we are reducing over heads
        # 2) for kv_seq_len, the splash attention prefetch schedule assumes no
        #     megacore
        # 3) for q_seq_len, we are reducing over it to compute dkv
        compiler_params=pltpu.CompilerParams(
            skip_device_barrier=cp_barrier,
            # Reserved for this context-only protocol; other communication
            # patterns (e.g. MoE) must use a different collective ID.
            collective_id=61440 if cp_barrier else None,
            dimension_semantics=("arbitrary",) * len(grid),
            flags={} if config.bwd_scheduler is None else {"XLA_TPU_FORCE_LP_LLO_SCHEDULER": config.bwd_scheduler},
            vmem_limit_bytes=config.bwd_vmem_limit_bytes,
            allow_input_fusion=allow_input_fusion,
        ),
        name="mla_fused_dq_dkv_a2a",
        cost_estimate=cost_estimate,
        interpret=config.interpret,
        metadata=metadata,
    )(*args, dq, dk, dv, *import_args)
    dq_unreduced, dk, dv, dq_final = results[:4]
  with jax.named_scope("splash_bwd_epilogue"):
    dq = dq_final
    if native_layout:
      dq = dq.mT
      dk = dk.mT
      dv = dv.mT
    dk = dk.astype(k.dtype)
    dv = dv.astype(v.dtype)
  if import_head is not None:
    return dq, dk, dv, *results[4:]
  return dq, dk, dv


@functools.lru_cache(maxsize=1)
def validate_source():
  actual = hashlib.sha256(Path(inspect.getfile(sk)).read_bytes()).hexdigest()
  if actual != SOURCE_SHA:
    raise ValueError(f"MLA fused backward requires Tokamax Splash {SOURCE_SHA}; got {actual}")


def make_backward(kernel, cp, *, axis_name="context", cp_barrier=False):
  validate_source()
  cfg = kernel.kwargs["config"]
  if (
      cp not in (8, 16)
      or kernel.kwargs["is_mqa"]
      or (cfg.block_q_dkv, cfg.block_kv_dkv, cfg.block_kv_dkv_compute, cfg.dq_reduction_steps) != (512, 4096, 512, 3)
      or not cfg.bwd_kv_unroll
      or cfg.bwd_scheduler is not None
      or not cfg.bwd_cast_before_transpose
      or cfg.bwd_dq_first
      or cfg.bwd_scale_after_dot
      or cfg.use_base2_exp
      or cfg.attn_logits_soft_cap is not None
      or any(layout != sk.QKVLayout.HEAD_DIM_MINOR for layout in (cfg.q_layout, cfg.k_layout, cfg.v_layout))
  ):
    raise ValueError("Unsupported MLA fused backward configuration")
  validate_metadata(kernel, cp)

  def backward(
      q,
      k,
      v,
      ids,
      o,
      lse,
      do,
      *,
      di=None,
      import_head=None,
      imports=None,
      kv_import_prefetch=None,
      kv_import_defer_send=False,
      kv_import_launch_after_credit=False,
      kv_hbm_prefetch=None,
      kv_hbm_trace=False,
      shadow_imports=None,
      shadow_prefetch=False,
  ):
    heads, sequence, _ = q.shape
    kv_import = kv_import_prefetch is not None or kv_hbm_prefetch is not None
    k_shape = (cp * heads, sequence // cp, 192) if kv_import else q.shape
    v_shape = (cp * heads, sequence // cp, 128) if kv_import else (heads, sequence, 128)
    if heads not in (1, 4) or k.shape != k_shape or q.shape != (heads, sequence, 192):
      raise ValueError("MLA backward requires one or four matching Q/K heads of width 192")
    if v.shape != v_shape or o.shape != (heads, sequence, 128) or do.shape != o.shape:
      raise ValueError("MLA backward V/O/dO must match the head count and sequence, with width 128")
    if ids.shape != (sequence,) or lse.shape != (heads, sequence):
      raise ValueError("MLA backward requires shared sequence IDs and per-head LSE")
    if heads == 1 and cp_barrier:
      raise ValueError("Per-head MLA backward requires the compiler global barrier")
    if di is None:
      with jax.named_scope("splash_bwd_di"):
        di = jnp.einsum("hsd,hsd->hs", o.astype(jnp.float32), do.astype(jnp.float32))
    elif di.shape != (heads, sequence) or di.dtype != jnp.float32:
      raise ValueError("Explicit MLA di must be FP32 with matching heads and sequence")
    return fused_backward(
        q,
        k,
        v,
        sk.SegmentIds(ids, ids),
        lse,
        do,
        di,
        cp=cp,
        axis_name=axis_name,
        cp_barrier=cp_barrier,
        import_head=import_head,
        imports=imports,
        kv_import_prefetch=kv_import_prefetch,
        kv_import_defer_send=kv_import_defer_send,
        kv_import_launch_after_credit=kv_import_launch_after_credit,
        kv_hbm_prefetch=kv_hbm_prefetch,
        kv_hbm_trace=kv_hbm_trace,
        shadow_imports=shadow_imports,
        shadow_prefetch=shadow_prefetch,
        bq=cfg.block_q_dkv,
        bkv=cfg.block_kv_dkv,
        bkv_compute=cfg.block_kv_dkv_compute,
        is_mqa=False,
        mask_info=kernel.dkv_mask_info,
        mask_value=kernel.kwargs["mask_value"],
        mask_function=kernel.kwargs["mask_function"],
        config=cfg,
        dkv_mask_sparsity=kernel.kwargs["dkv_mask_sparsity"],
    )

  return backward
