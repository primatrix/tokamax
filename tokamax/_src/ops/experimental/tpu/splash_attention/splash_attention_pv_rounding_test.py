"""PV-only ablation: natural exp, legacy score scale, unrounded denominator."""
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as splash
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as masks


def config(width):
  return splash.SplashConfig(
      block_q=128, block_kv=128, block_kv_compute=128,
      block_q_dkv=128, block_kv_dkv=128, interpret=True,
      use_base2_exp=False, combine_log2_scale=False,
      softmax_scale=width**-0.5, pv_bf16_rounding=True,
  )


@pytest.mark.parametrize('width,length', [(72,128), (72,256), (128,256)])
def test_natural_exp_pv_rounding_and_fp32_denominator(width, length):
  q,k,v = [jax.random.normal(key,(1,length,width),jnp.bfloat16)
           for key in jax.random.split(jax.random.key(42),3)]
  cfg=config(width)
  mask=masks.FullMask((length,length))
  actual,stats=splash.make_splash_mha_single_device(mask,config=cfg,save_residuals=True)(q,k,v)
  _,old_stats=splash.make_splash_mha_single_device(
      mask,config=dataclasses.replace(cfg,pv_bf16_rounding=False),save_residuals=True)(q,k,v)
  # Rounding the PV operand must not alter the softmax denominator/statistics.
  np.testing.assert_array_equal(stats['logsumexp'],old_stats['logsumexp'])
  qf,kf,vf=(x.astype(jnp.float32) for x in (q,k,v))
  m=jnp.full((1,length,1),-jnp.inf);denom=jnp.zeros_like(m);num=jnp.zeros_like(qf)
  for start in range(0,length,128):
    scores=jnp.einsum('hqd,hkd->hqk',qf,kf[:,start:start+128])*jnp.float32(width**-0.5)
    new_m=jnp.maximum(m,scores.max(axis=-1,keepdims=True))
    p=jnp.exp(scores-new_m);alpha=jnp.exp(m-new_m)
    num=alpha*num+jnp.einsum('hqk,hkd->hqd',p.astype(jnp.bfloat16).astype(jnp.float32),vf[:,start:start+128])
    denom=alpha*denom+p.sum(axis=-1,keepdims=True);m=new_m
  np.testing.assert_allclose(actual.astype(jnp.float32),(num/denom).astype(jnp.bfloat16).astype(jnp.float32),atol=2e-3,rtol=1e-2)
  np.testing.assert_allclose(stats['logsumexp'],(m+jnp.log(denom))[...,0],atol=2e-5,rtol=2e-5)


def test_backward_against_dense_reference_with_causal_mask():
  q,k,v,do=[jax.random.normal(key,(1,128,72),jnp.bfloat16)*.25
            for key in jax.random.split(jax.random.key(9),4)]
  allowed=np.tril(np.ones((128,128),dtype=bool))
  fn=splash.make_splash_mha_single_device(masks.NumpyMask(allowed),config=config(72))
  out,pb=jax.vjp(fn,q,k,v)
  def dense(q,k,v):
    scores=jnp.einsum('hqd,hkd->hqk',q.astype(jnp.float32),k.astype(jnp.float32))*72**-.5
    p=jax.nn.softmax(jnp.where(allowed,scores,-jnp.inf),axis=-1)
    return jnp.einsum('hqk,hkd->hqd',p,v.astype(jnp.float32))
  _,ref_pb=jax.vjp(dense,q,k,v)
  assert bool(jnp.isfinite(out).all())
  for actual,expected in zip(pb(do),ref_pb(do.astype(jnp.float32))):
    assert bool(jnp.isfinite(actual).all())
    np.testing.assert_allclose(actual.astype(jnp.float32),expected.astype(jnp.float32),atol=2e-3,rtol=3e-2)


def test_reject_fp32_inputs():
  fn=splash.make_splash_mha_single_device(masks.FullMask((128,128)),config=config(72))
  x=jnp.zeros((1,128,72),jnp.float32)
  with pytest.raises(ValueError,match='BF16'):
    fn(x,x,x)
