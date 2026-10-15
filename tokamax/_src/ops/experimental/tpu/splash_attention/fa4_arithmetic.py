"""Contractible FP32 multiply-add request for the experimental FA4 arithmetic path."""

from jax._src import core
from jax._src.interpreters import ad, batching, mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.pallas.mosaic import lowering
import jax.numpy as jnp

_fma_p = core.Primitive("tokamax_fa4_fma")
_fma_p.def_abstract_eval(lambda a, b, c: a)
_fma_p.def_impl(lambda a, b, c: a * b + c)
# CPU Pallas interpretation is an arithmetic check, not a fused-instruction
# parity oracle. The native TPU lowering permits contraction of mul/add.
# This is not a guarantee of a fused instruction or single rounding.
mlir.register_lowering(
    _fma_p, mlir.lower_fun(lambda a, b, c: a * b + c, multiple_results=False)
)


@lowering.register_lowering_rule(_fma_p)
def _fma_lowering(ctx, a, b, c):
  del ctx
  # The pinned Mosaic backend cannot legalize vector math.fma. Use its
  # supported arithmetic ops, permitting contraction without reassociation,
  # finite-only assumptions, or other fast-math relaxations.
  contract = ir.Attribute.parse("#arith.fastmath<contract>")
  product = arith.mulf(a, b, fastmath=contract)
  return arith.addf(product, c, fastmath=contract)


def _batch(args, dims):
  size = next(a.shape[d] for a, d in zip(args, dims) if d is not None)
  args = [batching.bdim_at_front(a, d, size) for a, d in zip(args, dims)]
  return _fma_p.bind(*args), 0


batching.primitive_batchers[_fma_p] = _batch
ad.defjvp(
    _fma_p,
    lambda t, a, b, c: t * b,
    lambda t, a, b, c: a * t,
    lambda t, a, b, c: t,
)


def fma(a, b, c):
  """Broadcast FP32 operands and permit native multiply-add contraction."""
  a, b, c = jnp.broadcast_arrays(
      *(jnp.asarray(x, jnp.float32) for x in (a, b, c))
  )
  return _fma_p.bind(a, b, c)
