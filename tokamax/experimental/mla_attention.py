"""Experimental sequence-layout MLA attention with fused TPU communication.

Supports the constrained BF16, CP8/CP16, four-local-head DQK192/DV128
contract described in docs/mla_attention.md. This API is experimental.
Model configuration, rematerialization policy and BF16 projection/gate
integration belong to the caller. Diagnostic exchange modes in the private
implementation are not additional production-qualified configurations.
"""

from tokamax._src.ops.experimental.tpu.splash_attention.mla_sequence_attention import make_attention as make_attention
from tokamax._src.ops.experimental.tpu.splash_attention.mla_sequence_attention import make_static_splash_kernel as make_static_splash_kernel

__all__ = ["make_attention", "make_static_splash_kernel"]
