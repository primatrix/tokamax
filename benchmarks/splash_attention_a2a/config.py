"""Frozen production CP8 contract used by the benchmark scripts."""

from __future__ import annotations


def production_config(*, global_heads: int = 64, cp_size: int = 8) -> dict:
  if global_heads % cp_size:
    raise ValueError("global_heads must be divisible by cp_size")
  return {
      "name": "ling3_mla_splash_attention_cp8",
      "cp_size": cp_size,
      "mesh_axis": "context",
      "global_heads": global_heads,
      "heads": global_heads // cp_size,
      "tokens": 262144,
      "qk_head_dim": 192,
      "v_head_dim": 128,
      "dtype": "bfloat16",
      "mask": "causal_with_runtime_segment_ids",
      "expected_active_blocks": 8256,
      "save_residuals": True,
      "splash": {
          "block_q": 2048,
          "block_kv": 2048,
          "block_kv_compute": 512,
          "block_q_dkv": 2048,
          "block_kv_dkv": 2048,
          "block_kv_dkv_compute": 512,
          "block_q_dq": None,
          "block_kv_dq": None,
          "use_fused_bwd_kernel": True,
          "q_layout": 2,
          "k_layout": 2,
          "v_layout": 1,
          "fwd_cost_estimate": None,
          "bwd_cost_estimate": None,
          "residual_checkpoint_name": "context",
          "attn_logits_soft_cap": None,
          "fuse_reciprocal": True,
          "use_base2_exp": False,
          "max_logit_const": None,
          "interpret": False,
          "dq_reduction_steps": 3,
          "combine_log2_scale": False,
          "bwd_kv_unroll": True,
          "bwd_dq_first": False,
          "bwd_cast_before_transpose": False,
          "bwd_scale_after_dot": False,
          "omit_unused_max_logits": False,
          "compact_stats_output": False,
          "bwd_scheduler": None,
          "segment_mask_on_partial_only": False,
          "fwd_vmem_limit_bytes": None,
          "bwd_vmem_limit_bytes": None,
          "use_experimental_scheduler": True,
          "softmax_scale": 0.07216878364870322,
      },
  }
