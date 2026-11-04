"""Experimental backward-only dO import and ordered Di cache.

Not production qualified. The standalone Di arithmetic is tested separately (exp-omwszlr3rl, source
169a6a9f633b5bdca2fc88651461ba43bcb201c8 in MaxText);
integrated DMA ordering, gradients and performance require device acceptance.
"""
import contextlib

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as sk


def di_tile(o, do):
  """Same ordered FP32 operations as the device-qualified standalone probe."""
  if o.shape != (512, 128) or do.shape != o.shape:
    raise ValueError("Expected two [512,128] Di tiles")
  if o.dtype != jnp.bfloat16 or do.dtype != jnp.bfloat16:
    raise ValueError("Di requires BF16 inputs")
  products = o.astype(jnp.float32) * do.astype(jnp.float32)
  p = products.T.reshape(16, 8, 512)
  acc = p[1] + p[0]
  for k in range(2, 16):
    acc = acc + p[k]
  for shift in (4, 2, 1):
    acc = pltpu.roll(acc, shift, axis=0) + acc
  return jnp.broadcast_to(acc[:1, :], (8, 512))


def do_hbm_output_kernel(*refs, base_kernel, cp, axis_name, heads, rows, bq, prefetch, trace_regions=False):
  """Import once per head/owner; compute Di in row 0, reuse it in later rows.

  Existing K/V wrapper receives its unchanged 48-ref ABI. Two new outputs
  precede its scratch: dO destination and Di cache. Nine new scratch refs are
  appended. Di write banks and cache-load banks are distinct halves of one
  four-slot allocation; their DMA semaphores remain separate.

  Two-slot dO/O reuse assumes ordered scratch loads versus subsequent DMA
  writes, as in the stable dQ input stage. This is not established by a dQ
  store fence: dV also reads dO. A deliberate same-slot overwrite control and
  repeated integrated gradient comparisons must precede acceptance.
  """
  assert len(refs) == 59 and heads == 4 and bq == 512 and rows % cp == 0
  original = list(refs[:26]) + list(refs[28:50])
  destination, di_cache = refs[26:28]
  do_bank, o_bank, di_bank, send, recv, do_load, o_load, di_store, di_load = refs[50:]
  source, o_source = original[12:14]
  head, program = pl.program_id(0), pl.program_id(1)
  programs = pl.num_programs(1)
  rank = lax.axis_index(axis_name)
  q_steps = rows * 8
  per_owner = q_steps // cp
  length = per_owner * bq
  slot = lax.rem(program, 2)

  def scope(name):
    return jax.named_scope(name) if trace_regions else contextlib.nullcontext()

  def address(index):
    return sk.to_i32(original[0][index]), sk.to_i32(original[1][index])

  row, col = address(program)

  def transfer(h, peer):
    return pltpu.make_async_remote_copy(
        source.at[peer * heads + h], destination.at[h, pl.ds(rank * length, length), :],
        send.at[h, peer], recv.at[h, rank], device_id={axis_name: peer})

  def launch(h):
    def push(offset, _):
      transfer(h, lax.rem(rank + offset, cp)).start()
    lax.fori_loop(0, cp, push, None, unroll=False)

  def receive(h, r, c):
    # Each counter is consumed exactly once. Later rows must never wait again.
    @pl.when(jnp.logical_and(r == 0, lax.rem(c, per_owner) == 0))
    def _():
      owner = lax.div(c, per_owner)
      with scope("mla_do_owner_recv"):
        pltpu.make_async_copy(source.at[rank * heads + h],
            destination.at[h, pl.ds(owner * length, length), :], recv.at[h, owner]).wait()

  def do_transfer(index):
    _, c = address(index)
    s = lax.rem(index, 2)
    return pltpu.make_async_copy(destination.at[head, pl.ds(c * bq, bq), :], do_bank.at[s], do_load.at[s])

  def o_transfer(index):
    _, c = address(index)
    s = lax.rem(index, 2)
    return pltpu.make_async_copy(o_source.at[head, pl.ds(c * bq, bq), :], o_bank.at[s], o_load.at[s])

  def di_write(index):
    _, c = address(index)
    s = lax.rem(index, 2)
    return pltpu.make_async_copy(di_bank.at[s], di_cache.at[head, :, pl.ds(c * bq, bq)], di_store.at[s])

  def di_read(index):
    _, c = address(index)
    s = lax.rem(index, 2)
    return pltpu.make_async_copy(di_cache.at[head, :, pl.ds(c * bq, bq)], di_bank.at[2 + s], di_load.at[s])

  @pl.when(program == 0)
  def bootstrap():
    if prefetch:
      @pl.when(head == 0)
      def _():
        launch(jnp.int32(0))
    else:
      launch(head)
    receive(head, row, col)
    do_transfer(program).start()
    o_transfer(program).start()

  with scope("mla_do_load_wait"):
    do_transfer(program).wait()

  @pl.when(row == 0)
  def compute_di():
    o_transfer(program).wait()
    @pl.when(program > 0)
    def _():
      di_write(program - 1).wait()
    with scope("mla_do_di_compute"):
      di_bank.at[slot][...] = di_tile(o_bank.at[slot][...], do_bank.at[slot][...])
    di_write(program).start()

  @pl.when(row > 0)
  def consume_cached_di():
    with scope("mla_do_di_cache_wait"):
      di_read(program).wait()

  @pl.when(program + 1 < programs)
  def prefetch_next():
    r, c = address(program + 1)
    receive(head, r, c)
    do_transfer(program + 1).start()
    @pl.when(r == 0)
    def _():
      o_transfer(program + 1).start()
    @pl.when(r > 0)
    def _():
      # Drain row0's final store before the first cache load is issued.
      @pl.when(row == 0)
      def _():
        di_write(program).wait()
      di_read(program + 1).start()

  original[12] = do_bank.at[slot]
  original[13] = di_bank.at[jnp.where(row == 0, slot, 2 + slot)]
  base_kernel(*original)

  if prefetch:
    @pl.when(jnp.logical_and(jnp.logical_and(row == 0, col == q_steps - 1), head + 1 < heads))
    def next_head():
      # Avoid adding next-head traffic during the current head's first scan.
      launch(head + 1)

  @pl.when(jnp.logical_and(head == heads - 1, program == programs - 1))
  def drain_sources():
    def wait_head(h, _):
      def wait_peer(peer, _):
        transfer(h, peer).wait_send()
      lax.fori_loop(0, cp, wait_peer, None, unroll=False)
    with scope("mla_do_send_drain"):
      lax.fori_loop(0, heads, wait_head, None, unroll=False)
