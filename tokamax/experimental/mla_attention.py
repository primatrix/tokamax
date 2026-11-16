"""Experimental sequence-layout MLA attention with fused TPU communication.

Supports BF16 causal MLA attention for any context-parallel size that evenly
partitions the backward K/V tiles (``block_kv_dkv`` tokens) and the global
heads, with head widths taken from the inputs, as described in
docs/mla_attention.md. This API is experimental. Model configuration,
rematerialization policy and BF16 projection/gate integration belong to the
caller.
"""

from tokamax._src.ops.experimental.tpu.splash_attention.mla_sequence_attention import make_attention as make_attention
from tokamax._src.ops.experimental.tpu.splash_attention.mla_sequence_attention import make_static_splash_kernel as make_static_splash_kernel

__all__ = ["make_attention", "make_static_splash_kernel"]
