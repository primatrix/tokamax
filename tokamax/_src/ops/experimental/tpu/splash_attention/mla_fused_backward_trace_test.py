"""Abstract regression of the actual fused caller at the production geometry.

Runs on CPU without allocating the large attention tensors. Device numerical
qualification is a separate CP8/CP16 test; this checks the production ABI.
"""

import dataclasses
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.ad_checkpoint import saved_residuals
from jax.sharding import Mesh
from tokamax._src.ops.experimental.tpu.splash_attention import mla_fused_backward
from tokamax._src.ops.experimental.tpu.splash_attention import mla_sequence_attention
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as sk
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as sm
from tokamax.experimental import mla_attention


@pytest.mark.skipif(len(jax.devices()) < 4, reason="Requires four logical devices")
@pytest.mark.parametrize("record_max_logits", [False, True])
def test_global_forward_boundary_preserves_values_and_vjp_ownership(record_max_logits):
  """Check the global VJP boundary against an independent elementwise function.

  A small analytic kernel makes every batch/head/token observable on CPU.
  Actual Splash arithmetic and remote DMA still require device qualification.
  """
  mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("fsdp", "context"))

  class AnalyticKernel:
    kwargs = {"config": SimpleNamespace(residual_checkpoint_name="mla_splash_residuals")}

    def __call__(self, q, k, v, ids, *, save_residuals):
      del ids
      assert save_residuals
      return q[..., :128] * k[..., :128] + v, {
          "logsumexp": jnp.zeros(q.shape[:2], jnp.float32),
          "max_logits": jnp.max(q.astype(jnp.float32), axis=-1),
      }

  def pullback(residuals, cotangent):
    q, k, _, _, _, _ = residuals
    do = cotangent[0] if record_max_logits else cotangent
    pad = ((0, 0), (0, 0), (0, 0), (0, 64))
    return jnp.pad(do * k[..., :128], pad), jnp.pad(do * q[..., :128], pad), do, None

  attention = mla_sequence_attention._make_attention_with_external_inputs(
      AnalyticKernel(),
      mesh=mesh,
      batch_axes="fsdp",
      axis_name="context",
      record_max_logits=record_max_logits,
      pullback=pullback,
      sequence_residuals=True,
      fwd_output_exchange="external",
  )
  shape = (2, 16, 8, 192)
  q = (jnp.arange(math.prod(shape)).reshape(shape) % 7).astype(jnp.bfloat16)
  k = ((jnp.arange(math.prod(shape)).reshape(shape) // 7) % 5).astype(jnp.bfloat16)
  v = (q[..., :128] % 3).astype(jnp.bfloat16)
  do = (k[..., :128] % 3).astype(jnp.bfloat16)
  ids = jnp.ones(shape[:2], jnp.int32)

  def reference(q, k, v):
    return q[..., :128] * k[..., :128] + v

  expected, reference_vjp = jax.vjp(reference, q, k, v)

  @jax.jit
  def cycle(q, k, v, ids, do):
    output, vjp = jax.vjp(lambda q, k, v: attention(q, k, v, ids), q, k, v)
    cotangent = (do, jnp.ones_like(output[1])) if record_max_logits else do
    return output, vjp(cotangent)

  with jax.set_mesh(mesh):
    actual, gradients = cycle(q, k, v, ids, do)
  if record_max_logits:
    np.testing.assert_array_equal(actual[1], jnp.max(q.astype(jnp.float32), axis=(1, 3)))
    actual = actual[0]
  np.testing.assert_array_equal(actual, expected)
  for gradient, reference_gradient in zip(gradients, reference_vjp(do), strict=True):
    np.testing.assert_array_equal(gradient, reference_gradient)


def config():
  return sk.SplashConfig(
      block_q=4096,
      block_kv=2048,
      block_kv_compute=512,
      block_q_dkv=512,
      block_kv_dkv=4096,
      block_kv_dkv_compute=512,
      use_fused_bwd_kernel=True,
      dq_reduction_steps=3,
      q_layout=sk.QKVLayout.HEAD_DIM_MINOR,
      k_layout=sk.QKVLayout.HEAD_DIM_MINOR,
      v_layout=sk.QKVLayout.HEAD_DIM_MINOR,
      bwd_cast_before_transpose=True,
      use_base2_exp=False,
      softmax_scale=0.07216878364870322,
      residual_checkpoint_name="mla_splash_residuals",
      use_experimental_scheduler=True,
  )


def _equations(value):
  if hasattr(value, "eqns"):
    for equation in value.eqns:
      yield equation
      yield from _equations(equation.params)
  elif hasattr(value, "jaxpr"):
    yield from _equations(value.jaxpr)
  elif isinstance(value, dict):
    for child in value.values():
      yield from _equations(child)
  elif isinstance(value, (tuple, list)):
    for child in value:
      yield from _equations(child)


def test_pinned_source_and_bad_config_rejection():
  mla_fused_backward.validate_source()
  cfg = config()
  kernel = sk.make_splash_mha(sm.CausalMask((32768, 32768)), config=cfg, q_seq_shards=1)
  bad = sk.SplashAttentionKernel(
      kernel.fwd_mask_info,
      kernel.dkv_mask_info,
      **dict(kernel.kwargs, config=dataclasses.replace(cfg, bwd_dq_first=True)),
  )
  with pytest.raises(ValueError, match="Unsupported MLA"):
    mla_fused_backward.make_backward(bad, 8)


@pytest.mark.parametrize("cp", [8, 16])
@pytest.mark.parametrize("rows_per_owner", [1, 2, 3, 4, 8])
def test_deferred_send_predecessors_match_independent_owner_history(cp, rows_per_owner):
  rows = cp * rows_per_owner
  generations = np.arange(4 * rows, dtype=np.int32)
  actual = np.asarray(mla_fused_backward.kv_previous_send(generations, rows=rows, cp=cp))
  history = {}
  drained = []
  for generation, previous in zip(generations, actual, strict=True):
    owner = (generation % rows) // rows_per_owner
    key = (int(owner), int(generation % 2))
    if key in history:
      assert previous == history[key]
      drained.append(int(previous))
    else:
      assert previous < 0
    history[key] = int(generation)
  # Every source counter has exactly one consuming wait, including final exit.
  assert sorted(drained + list(history.values())) == list(generations)
  assert len(history) == cp * min(2, rows_per_owner)


@pytest.mark.parametrize("rows,cp", [(3, 3), (9, 3), (10, 8), (4, 8), (16, 0)])
def test_deferred_send_predecessors_reject_invalid_partitions(rows, cp):
  with pytest.raises(ValueError, match="even rows and complete owner partitions"):
    mla_fused_backward.kv_previous_send(jnp.int32(0), rows=rows, cp=cp)


def test_static_mask_metadata_stays_numpy_with_outer_mesh():
  mesh = Mesh(np.asarray(jax.devices()), ("outer",))
  observed = []
  reference = sk.make_splash_mha(sm.CausalMask((32768, 32768)), config=config(), q_seq_shards=1)
  expected = jax.tree.leaves((reference.fwd_mask_info, reference.dkv_mask_info))

  def traced(x):
    kernel = mla_sequence_attention.make_static_splash_kernel(sm.CausalMask((32768, 32768)), config())
    leaves = jax.tree.leaves((kernel.fwd_mask_info, kernel.dkv_mask_info))
    assert leaves
    for leaf, original in zip(leaves, expected, strict=True):
      assert isinstance(leaf, np.ndarray)
      np.testing.assert_array_equal(leaf, original)
    with jax.ensure_compile_time_eval():
      observed.append(jax.sharding.get_abstract_mesh())
    return x

  with jax.set_mesh(mesh):
    jax.eval_shape(traced, jax.ShapeDtypeStruct((), jnp.float32))
    assert jax.sharding.get_abstract_mesh() == mesh.abstract_mesh
  assert observed == [mesh.abstract_mesh]


@pytest.fixture(scope="module")
def export_metadata_kernel():
  return mla_sequence_attention.make_static_splash_kernel(sm.CausalMask((65536, 65536)), config())


@pytest.mark.parametrize("cp", [8, 16])
def test_export_writes_match_actual_bounds(export_metadata_kernel, cp):
  kernel = export_metadata_kernel
  proof = mla_fused_backward.validate_metadata(kernel, cp)
  info = kernel.dkv_mask_info
  n = proof["programs_per_head"]
  _, actual_end = sk.mask_info_lib.find_bounds(info.active_rows)
  writes = np.asarray(actual_end)[:n].astype(bool)
  np.testing.assert_array_equal(np.asarray(info.active_rows)[:n][writes], np.arange(proof["kv_steps"]))
  assert proof["export_writes_per_head"] == proof["kv_steps"]
  assert np.count_nonzero(writes) < n  # Interior Q accumulation does not send.


@pytest.mark.parametrize("cp", [8, 16])
@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("owner_first", [False, True])
def test_q_receive_precedes_actual_local_dma_start(export_metadata_kernel, monkeypatch, cp, prefetch, owner_first):
  """Execute the wrapper's actual callbacks with a protocol-checking DMA stub.

  This checks issue/wait order, not hardware visibility or arithmetic. In
  particular a wait in prepare(g) is too late for load(g) issued at g-1.
  """
  kernel = export_metadata_kernel
  proof = mla_fused_backward.validate_metadata(kernel, cp)
  n, bq, sequence = proof["programs_per_head"], 512, 65536
  rows = np.asarray(kernel.dkv_mask_info.active_rows, dtype=np.int32)
  cols = np.asarray(kernel.dkv_mask_info.active_cols, dtype=np.int32)
  assert proof["q_steps"] * bq == sequence
  length = sequence // cp
  assert np.all(cols[:n] * bq // length == ((cols[:n] + 1) * bq - 1) // length)
  issued, received, loaded = set(), set(), set()
  launches, receive_programs = {}, {}
  current = [0, 0]

  class Ref:
    def __init__(self, name, indices=()):
      self.name, self.indices = name, indices

    @property
    def at(self):
      return self

    def __getitem__(self, indices):
      return Ref(self.name, indices)

  class Transfer:
    def __init__(self, source, destination, semaphore, remote=False):
      self.source, self.destination, self.semaphore, self.remote = source, destination, semaphore, remote

    def start(self):
      head, program = current
      if self.remote:
        target_head, peer = map(int, self.semaphore.indices)
        if target_head > head:
          assert {(head, o) for o in range(cp)} <= received
        issued.add((target_head, peer))
        launches.setdefault(target_head, set()).add((head, program))
      else:
        assert self.destination.name == "bank"
        generation = program if program == 0 and (head, 0) not in loaded else program + 1
        tile = self.source.indices[1].start // bq
        assert tile == int(cols[generation])
        assert int(self.destination.indices) == generation % 3
        assert (head, tile * bq // length) in received, "local Q DMA started before owner receive"
        loaded.add((head, generation))

    def wait(self):
      head, program = current
      if self.semaphore.name == "recv":
        key = tuple(map(int, self.semaphore.indices))
        assert key not in received, "receive consumed twice"
        assert key[0] == head and {(head, p) for p in range(cp)} <= issued
        received.add(key)
        receive_programs[key] = program
      else:
        assert (head, program) in loaded

    def wait_send(self):
      assert tuple(map(int, self.semaphore.indices)) in issued

  def when(condition):
    def invoke(fn):
      if bool(condition):
        fn()
    return invoke

  def fori_loop(start, stop, body, carry, **kwargs):
    del kwargs
    for value in range(int(start), int(stop)):
      carry = body(np.int32(value), carry)
    return carry

  monkeypatch.setattr(mla_fused_backward, "pl", SimpleNamespace(
      program_id=lambda axis: np.int32(current[axis]), num_programs=lambda axis: n,
      when=when, ds=lambda start, size: slice(int(start), int(start + size)),
  ))
  monkeypatch.setattr(mla_fused_backward, "lax", SimpleNamespace(
      axis_index=lambda axis: np.int32(0), rem=np.remainder, div=np.floor_divide, fori_loop=fori_loop,
  ))
  monkeypatch.setattr(mla_fused_backward, "pltpu", SimpleNamespace(
      make_async_copy=lambda src, dst, sem: Transfer(src, dst, sem),
      make_async_remote_copy=lambda src, dst, send, recv, **kwargs: Transfer(src, dst, send, remote=True),
  ))
  monkeypatch.setattr(mla_fused_backward, "jnp", SimpleNamespace(int32=np.int32, logical_and=np.logical_and))
  monkeypatch.setattr(mla_fused_backward, "sk", SimpleNamespace(to_i32=np.int32))
  refs = [Ref(str(i)) for i in range(53)]
  refs[:2] = rows, cols
  refs[49:] = [Ref(name) for name in ("bank", "send", "recv", "load")]

  def body(*args, q_prepare, q_prefetch):
    del args
    q_prepare()
    q_prefetch()

  for head in range(4):
    for program in range(n):
      current[:] = head, program
      mla_fused_backward.q_hbm_output_kernel(
          *refs, base_kernel=body, cp=cp, axis_name="context", heads=4,
          sequence=sequence, bq=bq, prefetch=prefetch, owner_first=owner_first,
      )
  assert received == {(h, o) for h in range(4) for o in range(cp)}
  assert loaded == {(h, g) for h in range(4) for g in range(n)}
  for head in range(4):
    for owner in range(cp):
      first = int(np.flatnonzero(cols[:n] * bq // length == owner)[0])
      assert receive_programs[head, owner] == (max(0, first - 1) if owner_first else 0)
    expected_launch = (head - 1, (cp - 1) * (length // bq) - 1 if owner_first else 0)
    assert launches[head] == {expected_launch if prefetch and head else (head, 0)}


def test_export_rejects_padding_that_suppresses_last_send(export_metadata_kernel):
  info = export_metadata_kernel.dkv_mask_info
  n = int(np.asarray(info.num_active_blocks)[0])
  # The dynamic grid ends before this repeated final row. find_bounds would
  # move its final write into padding, leaving the final send/drain unmatched.
  padded = info._replace(
      active_rows=np.insert(info.active_rows, n, info.active_rows[n - 1]),
      active_cols=np.insert(info.active_cols, n, info.active_cols[n - 1]),
  )
  with pytest.raises(ValueError, match="exactly one final write"):
    mla_fused_backward.validate_metadata(SimpleNamespace(dkv_mask_info=padded), 16)


@pytest.mark.parametrize("count", [0, -1, 1000000])
def test_export_rejects_invalid_active_count(export_metadata_kernel, count):
  info = export_metadata_kernel.dkv_mask_info._replace(num_active_blocks=np.asarray([count], dtype=np.int32))
  with pytest.raises(ValueError, match="active program count"):
    mla_fused_backward.validate_metadata(SimpleNamespace(dkv_mask_info=info), 16)


def test_export_rejects_reordered_accumulation(export_metadata_kernel):
  info = export_metadata_kernel.dkv_mask_info
  cols = np.array(info.active_cols, copy=True)
  cols[[0, 1]] = cols[[1, 0]]
  with pytest.raises(ValueError, match="ordered causal tile schedule"):
    mla_fused_backward.validate_metadata(SimpleNamespace(dkv_mask_info=info._replace(active_cols=cols)), 16)


@pytest.mark.skipif(len(jax.devices()) < 16, reason="Run with 16 logical CPU devices or on a 16-device slice")
@pytest.mark.parametrize("cp", [8, 16])
@pytest.mark.parametrize("cp_barrier", [False, True])
@pytest.mark.parametrize(
    "bwd_input_exchange",
    [
        "remat",
        "packed",
        "separate",
        "remote_dma",
        "per_head_split",
        "per_head_a2a",
        "per_head_import",
        "kv_credit_serial",
        "kv_credit_prefetch",
        "kv_credit_deferred",
        "kv_credit_ahead",
        "kv_hbm_serial",
        "kv_hbm_prefetch",
        "kv_hbm_traced",
        "qkv_hbm_serial",
        "qkv_hbm_prefetch",
        "qkv_hbm_owner_first",
        "kv_shadow_serial",
        "kv_shadow_prefetch",
    ],
)
def test_full_sequence_vjp_output_ownership(cp, cp_barrier, bwd_input_exchange):
  if cp_barrier and bwd_input_exchange.startswith(("per_head", "kv_credit", "kv_hbm", "qkv_hbm", "kv_shadow")):
    pytest.skip("Experimental import paths require the compiler global barrier")
  fsdp, heads = 16 // cp, cp * 4
  mesh = Mesh(np.asarray(jax.devices()[:16]).reshape(fsdp, cp), ("fsdp", "context"))
  seq = 262144
  with jax.ensure_compile_time_eval():
    kernel = mla_sequence_attention.make_static_splash_kernel(sm.CausalMask((seq, seq)), config())
    attention = mla_attention.make_attention(
        kernel,
        mesh=mesh,
        batch_axes="fsdp",
        record_max_logits=True,
        cp_barrier=cp_barrier,
        bwd_input_exchange=bwd_input_exchange,
    )

  def cycle(q, k, v, ids, do):
    (out, stats), pullback = jax.vjp(lambda q, k, v: attention(q, k, v, ids), q, k, v)
    return out, stats, *pullback((do, jnp.ones_like(stats)))

  args = [jax.ShapeDtypeStruct((fsdp, seq, heads, d), jnp.bfloat16) for d in (192, 192, 128)]
  args += [jax.ShapeDtypeStruct((fsdp, seq), jnp.int32), jax.ShapeDtypeStruct((fsdp, seq, heads, 128), jnp.bfloat16)]
  with jax.set_mesh(mesh):
    program = jax.make_jaxpr(cycle)(*args)
  assert [v.aval.shape for v in program.jaxpr.outvars] == [
      (fsdp, seq, heads, 128),
      (fsdp, heads),
      (fsdp, seq, heads, 192),
      (fsdp, seq, heads, 192),
      (fsdp, seq, heads, 128),
  ]
  assert "name=mla_fused_dq_dkv_a2a" in str(program)
  if bwd_input_exchange.startswith(("kv_hbm", "qkv_hbm")):
    # Forward retains the original global sharding constraints. The only
    # explicit all_to_all operations here belong to the backward pullback.
    widths = sorted(
        equation.invars[0].aval.shape[-1]
        for equation in _equations(program)
        if equation.primitive.name == "all_to_all"
    )
    assert widths == ([128] if bwd_input_exchange.startswith("qkv_hbm") else [128, 192])


@pytest.mark.skipif(len(jax.devices()) < 16, reason="Requires 16 logical devices")
@pytest.mark.parametrize("cp", [8, 16])
def test_input_exchanges_preserve_original_policy_saved_tensors(export_metadata_kernel, cp):
  fsdp, seq = 16 // cp, 65536
  mesh = Mesh(np.asarray(jax.devices()[:16]).reshape(fsdp, cp), ("fsdp", "context"))
  args = [jax.ShapeDtypeStruct((fsdp, seq, cp * 4, d), jnp.bfloat16) for d in (192, 192, 128)]
  args.append(jax.ShapeDtypeStruct((fsdp, seq), jnp.int32))
  saved_sets = []
  for exchange in (
      "remat",
      "packed",
      "separate",
      "remote_dma",
      "per_head_split",
      "per_head_a2a",
      "per_head_import",
      "kv_credit_serial",
      "kv_credit_prefetch",
      "kv_credit_deferred",
      "kv_credit_ahead",
      "kv_hbm_serial",
      "kv_hbm_prefetch",
      "kv_hbm_traced",
      "qkv_hbm_serial",
      "qkv_hbm_prefetch",
      "qkv_hbm_owner_first",
      "kv_shadow_serial",
      "kv_shadow_prefetch",
  ):
    attention = mla_attention.make_attention(
        export_metadata_kernel,
        mesh=mesh,
        batch_axes="fsdp",
        record_max_logits=True,
        bwd_input_exchange=exchange,
    )
    checkpointed = jax.checkpoint(
        attention,
        prevent_cse=True,
        policy=jax.checkpoint_policies.save_only_these_names("mla_splash_residuals"),
    )
    with jax.set_mesh(mesh):
      residuals = saved_residuals(checkpointed, *args)
    saved_sets.append(
        sorted(
            (str(aval.dtype), math.prod(aval.shape), origin if origin.startswith("from the argument") else "activation")
            for aval, origin in residuals
            if origin != "from a constant"
        )
    )
  assert all(saved == saved_sets[0] for saved in saved_sets)
  expected = sorted(
      [
          ("bfloat16", fsdp * seq * cp * 4 * 192, "from the argument q"),
          ("bfloat16", fsdp * seq * cp * 4 * 192, "from the argument k"),
          ("bfloat16", fsdp * seq * cp * 4 * 128, "from the argument v"),
          ("int32", fsdp * seq, "from the argument ids"),
          ("bfloat16", 16 * 4 * seq * 128, "activation"),
          ("float32", 16 * 4 * seq, "activation"),
      ]
  )
  assert all(saved == expected for saved in saved_sets)


@pytest.mark.parametrize(
    "exchange",
    [
        "packed",
        "separate",
        "remote_dma",
        "per_head_split",
        "per_head_a2a",
        "per_head_import",
        "kv_credit_serial",
        "kv_credit_prefetch",
        "kv_credit_deferred",
        "kv_credit_ahead",
        "kv_hbm_serial",
        "kv_hbm_prefetch",
        "kv_hbm_traced",
        "qkv_hbm_serial",
        "qkv_hbm_prefetch",
        "qkv_hbm_owner_first",
        "kv_shadow_serial",
        "kv_shadow_prefetch",
    ],
)
def test_experimental_inputs_reject_ordinary_vjp(export_metadata_kernel, exchange):
  with pytest.raises(ValueError, match="require the fused VJP"):
    mla_sequence_attention.make_local_attention(export_metadata_kernel, cp=8, fused=False, bwd_input_exchange=exchange)


@pytest.mark.parametrize(
    "exchange",
    [
        "per_head_split",
        "per_head_a2a",
        "per_head_import",
        "kv_credit_serial",
        "kv_credit_prefetch",
        "kv_credit_deferred",
        "kv_credit_ahead",
        "kv_hbm_serial",
        "kv_hbm_prefetch",
        "kv_hbm_traced",
        "qkv_hbm_serial",
        "qkv_hbm_prefetch",
        "qkv_hbm_owner_first",
        "kv_shadow_serial",
        "kv_shadow_prefetch",
    ],
)
def test_per_head_inputs_reject_cp_barrier(export_metadata_kernel, exchange):
  with pytest.raises(ValueError, match="global barrier"):
    mla_sequence_attention.make_local_attention(
        export_metadata_kernel,
        cp=8,
        cp_barrier=True,
        bwd_input_exchange=exchange,
    )


def test_unknown_input_exchange_rejected(export_metadata_kernel):
  with pytest.raises(ValueError, match="Unsupported MLA backward input exchange"):
    mla_sequence_attention.make_local_attention(export_metadata_kernel, cp=8, bwd_input_exchange="invalid")


def test_unknown_forward_input_exchange_rejected(export_metadata_kernel):
  with pytest.raises(ValueError, match="Unsupported MLA forward input exchange"):
    mla_sequence_attention.make_local_attention(export_metadata_kernel, cp=8, fwd_input_exchange="invalid")


@pytest.mark.skipif(len(jax.devices()) < 16, reason="Requires 16 logical devices")
@pytest.mark.parametrize("cp", [8, 16])
def test_per_head_exchange_and_gradient_values(cp):
  from jax import lax
  from jax.sharding import PartitionSpec as P

  fsdp, length = 16 // cp, 4
  mesh = Mesh(np.asarray(jax.devices()[:16]).reshape(fsdp, cp), ("fsdp", "context"))
  for tensor, width in enumerate((192, 192, 128, 128)):
    shape = (fsdp, cp, cp * 4, length, width)
    values = np.arange(math.prod(shape), dtype=np.int32).reshape(shape) + tensor * 10_000_000

    def local(x):
      x = x[0, 0]
      full = lax.all_to_all(x, "context", split_axis=0, concat_axis=1, tiled=True)
      heads = tuple(mla_sequence_attention._exchange_head(x, h, cp, "context") for h in range(4))
      parts = tuple(lax.all_to_all(a, "context", split_axis=1, concat_axis=0, tiled=True) for a in heads)
      assembled = mla_sequence_attention._merge_head_gradients(parts)
      return full[None, None], jnp.concatenate(heads, axis=0)[None, None], assembled[None, None]

    fn = jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=P("fsdp", "context"),
            out_specs=(P("fsdp", "context"),) * 3,
            check_vma=False,
        )
    )
    full, heads, roundtrip = map(np.asarray, fn(jnp.asarray(values)))
    np.testing.assert_array_equal(full, heads)
    np.testing.assert_array_equal(roundtrip, values)
  # Distinguish every sender/head pair independently of communication roundtrip.
  parts = tuple(jnp.broadcast_to(jnp.arange(cp)[:, None, None] * 4 + h, (cp, length, 3)) for h in range(4))
  merged = np.asarray(mla_sequence_attention._merge_head_gradients(parts))
  np.testing.assert_array_equal(merged, np.broadcast_to(np.arange(cp * 4)[:, None, None], merged.shape))


@pytest.mark.parametrize(
    "bad_di", [jax.ShapeDtypeStruct((1, 65536), jnp.bfloat16), jax.ShapeDtypeStruct((2, 65536), jnp.float32)]
)
def test_explicit_di_rejected(export_metadata_kernel, bad_di):
  backward = mla_fused_backward.make_backward(export_metadata_kernel, 16)
  q = jax.ShapeDtypeStruct((1, 65536, 192), jnp.bfloat16)
  v = jax.ShapeDtypeStruct((1, 65536, 128), jnp.bfloat16)
  with pytest.raises(ValueError, match="Explicit MLA di"):
    backward(
        q, q, v, jax.ShapeDtypeStruct((65536,), jnp.int32), v, jax.ShapeDtypeStruct((1, 65536), jnp.float32), v, di=bad_di
    )


def test_two_head_backward_rejected(export_metadata_kernel):
  backward = mla_fused_backward.make_backward(export_metadata_kernel, 16)
  q = jax.ShapeDtypeStruct((2, 65536, 192), jnp.bfloat16)
  v = jax.ShapeDtypeStruct((2, 65536, 128), jnp.bfloat16)
  with pytest.raises(ValueError, match="one or four"):
    backward(q, q, v, jax.ShapeDtypeStruct((65536,), jnp.int32), v, jax.ShapeDtypeStruct((2, 65536), jnp.float32), v)


@pytest.mark.parametrize(
    "case",
    [
        "head_zero",
        "head_four",
        "bool_head",
        "four_heads",
        "missing_sources",
        "short_sources",
        "wrong_shape",
        "wrong_dtype",
        "sources_without_head",
        "cp_barrier",
    ],
)
def test_import_rejects_invalid_contract(export_metadata_kernel, case):
  cp, sequence = 16, 65536
  heads = 4 if case == "four_heads" else 1
  shape = lambda dims, dtype=jnp.bfloat16: jax.ShapeDtypeStruct(dims, dtype)
  q, v = shape((heads, sequence, 192)), shape((heads, sequence, 128))
  sources = tuple(shape((cp * 4, sequence // cp, d)) for d in (192, 192, 128, 128))
  import_head = {"head_zero": 0, "head_four": 4, "bool_head": True, "sources_without_head": None}.get(case, 1)
  if case == "missing_sources":
    sources = None
  elif case == "short_sources":
    sources = sources[:3]
  elif case == "wrong_shape":
    sources = (shape((cp * 4, sequence // cp, 128)), *sources[1:])
  elif case == "wrong_dtype":
    sources = (shape(sources[0].shape, jnp.float32), *sources[1:])
  backward = mla_fused_backward.make_backward(export_metadata_kernel, cp, cp_barrier=case == "cp_barrier")
  with pytest.raises(ValueError, match="[Ii]mport|global barrier"):
    backward(
        q,
        q,
        v,
        shape((sequence,), jnp.int32),
        v,
        shape((heads, sequence), jnp.float32),
        v,
        di=shape((heads, sequence), jnp.float32),
        import_head=import_head,
        imports=sources,
    )
