"""Static and boundary checks for the opt-in forward output DMA path."""

import inspect

import jax.numpy as jnp
import pytest


from tokamax._src.ops.experimental.tpu.splash_attention import mla_sequence_attention as module


def test_forward_output_exchange_shape_contract(monkeypatch):
  def fake_pallas_call(*args, **kwargs):
    del args
    shape = kwargs["out_shape"]
    return lambda x: jnp.zeros(shape.shape, dtype=shape.dtype)

  monkeypatch.setattr(module.pl, "pallas_call", fake_pallas_call)
  output = jnp.zeros((4, 32768, 128), jnp.bfloat16)
  assert module._forward_output_exchange(output, cp=8, axis_name="context").shape == (32, 4096, 128)


def test_forward_input_exchange_shape_contract(monkeypatch):
  def fake_pallas_call(*args, **kwargs):
    del args
    shape = kwargs["out_shape"]
    return lambda x: jnp.zeros(shape.shape, dtype=shape.dtype)

  monkeypatch.setattr(module.pl, "pallas_call", fake_pallas_call)
  source = jnp.zeros((32, 4096, 512), jnp.bfloat16)
  assert module._forward_input_exchange(source, cp=8, axis_name="context").shape == (4, 32768, 512)
  source = jnp.zeros((32, 4096, 640), jnp.bfloat16)
  assert module._backward_input_exchange(source, cp=8, axis_name="context").shape == (4, 32768, 640)


@pytest.mark.parametrize("shape", [(31, 4096, 512), (32, 4097, 512), (32, 4096, 512)])
def test_forward_input_exchange_rejects_unqualified_geometry(monkeypatch, shape):
  monkeypatch.setattr(module.pl, "pallas_call", lambda *args, **kwargs: lambda x: x)
  dtype = jnp.float32 if shape == (32, 4096, 512) else jnp.bfloat16
  with pytest.raises(ValueError):
    module._forward_input_exchange(jnp.zeros(shape, dtype), cp=8, axis_name="context")


@pytest.mark.parametrize("shape", [(3, 32768, 128), (4, 32769, 128), (4, 32768, 127)])
def test_forward_output_exchange_rejects_unqualified_geometry(monkeypatch, shape):
  monkeypatch.setattr(module.pl, "pallas_call", lambda *args, **kwargs: lambda x: x)
  with pytest.raises(ValueError):
    module._forward_output_exchange(jnp.zeros(shape, jnp.bfloat16), cp=8, axis_name="context")


@pytest.mark.parametrize(
    "cp, dtype",
    [(4, jnp.bfloat16), (8, jnp.float32), (16, jnp.float16)],
)
def test_forward_output_exchange_rejects_unqualified_cp_or_dtype(monkeypatch, cp, dtype):
  monkeypatch.setattr(module.pl, "pallas_call", lambda *args, **kwargs: lambda x: x)
  with pytest.raises(ValueError):
    module._forward_output_exchange(jnp.zeros((4, 32768, 128), dtype), cp=cp, axis_name="context")


def test_forward_output_kernel_uses_remote_dma_and_drains():
  source = inspect.getsource(module)
  assert "mla_fwd_input_remote_a2a" in source
  assert "mla_bwd_input_remote_a2a" in source
  assert "mla_fwd_output_remote_a2a" in source
  assert "make_async_remote_copy" in source
  assert "wait_send()" in source
  assert "recv_ref.at[source, source_head, local]" in source
  assert "output_ref.at[sender * 4 + head" in source
  assert "pl.ds(sender * 4 + head, 1)" not in source
