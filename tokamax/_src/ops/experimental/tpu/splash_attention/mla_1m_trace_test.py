"""Trace the running Tiny-VL geometry without allocating its Q/K/V arrays."""
import dataclasses
import jax
import jax.numpy as jnp
import pytest
from tokamax._src.ops.experimental.tpu.splash_attention import mla_a2a_cp_backward_trace_test as t


@pytest.mark.skipif(len(jax.devices())<16,reason="Needs16 virtual CPU devices")
def test_1m_cp16_single_head_forward_backward_and_peer_barrier(monkeypatch):
  cfg=dataclasses.replace(t.config(),block_q=4096,block_kv=8192,block_kv_compute=1024,
      block_q_dkv=4096,block_kv_dkv=4096,block_kv_dkv_compute=1024,fwd_kv_unroll=False)
  monkeypatch.setattr(t,"CP",16);monkeypatch.setattr(t,"HEADS",16);monkeypatch.setattr(t,"SEQ",1048576)
  monkeypatch.setattr(t,"config",lambda:cfg)
  original=t.a2a.make_splash_mha_a2a_cp
  monkeypatch.setattr(t.a2a,"make_splash_mha_a2a_cp",lambda *a,**k:original(*a,**k,cp_barrier=True,segment_schedule=True))
  fn=t.attention("kv_hbm_prefetch",None,(t.RESIDUALS,))
  args=t.inputs()+[jax.ShapeDtypeStruct((16,1048576,128),jnp.bfloat16)]
  closed=jax.make_jaxpr(t.cycle(fn))(*args)
  assert [v.aval.shape for v in closed.jaxpr.outvars]==[
      (16,1048576,128),(16,1048576,192),(16,1048576,192),(16,1048576,128)]
