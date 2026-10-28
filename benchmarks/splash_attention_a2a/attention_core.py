"""Direct Tokamax Splash call for the Ling3 MLA attention-core HLO.

No projection, RoPE, RMSNorm, collective, backward computation or handwritten
attention. Only small mask metadata is instantiated; Q/K/V are abstract inputs
for CPU-hosted TPU AOT.
"""
import dataclasses
import hashlib
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel as sk
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as sm


def build(config):
    h, n = config["heads"], config["tokens"]
    dq, dv = config["qk_head_dim"], config["v_head_dim"]
    if config["mask"] != "causal_with_runtime_segment_ids":
        raise ValueError("This fixture reproduces causal segmented forward only")
    if config["dtype"] != "bfloat16" or not config["save_residuals"]:
        raise ValueError("The supplied custom-call requires BF16 and both residuals")
    fields = {f.name for f in dataclasses.fields(sk.SplashConfig)}
    unknown = config["splash"].keys() - fields
    if unknown:
        raise ValueError(f"Wrong Tokamax revision: unsupported config fields {unknown}")
    splash_config = sk.SplashConfig(**config["splash"])
    kernel = sk.make_splash_mha(
        sm.CausalMask((n, n)), config=splash_config, q_seq_shards=1,
    )
    size = kernel.fwd_mask_info.active_rows.size
    if size != config["expected_active_blocks"]:
        raise ValueError(f"Active grid mismatch: {size} != {config['expected_active_blocks']}")

    def core(q, k, v, q_segment_ids, kv_segment_ids):
        return kernel(q, k, v, sk.SegmentIds(q_segment_ids, kv_segment_ids),
                      save_residuals=True)

    specs = (
        jax.ShapeDtypeStruct((h, n, dq), jnp.bfloat16),
        jax.ShapeDtypeStruct((h, n, dq), jnp.bfloat16),
        jax.ShapeDtypeStruct((h, n, dv), jnp.bfloat16),
        jax.ShapeDtypeStruct((n,), jnp.int32),
        jax.ShapeDtypeStruct((n,), jnp.int32),
    )
    metadata = {}
    for name in kernel.fwd_mask_info._fields:
        value = getattr(kernel.fwd_mask_info, name)
        if value is not None:
            a = np.asarray(value)
            metadata[name] = dict(shape=list(a.shape), dtype=str(a.dtype),
                                  sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                                  values=a.tolist())
    source = Path(inspect.getsourcefile(sk))
    return core, specs, dict(
        tokamax_source=str(source),
        tokamax_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        actual_splash_config=dataclasses.asdict(splash_config), mask_metadata=metadata,
        mask_assumption="Causal mask from resolved decoder MLA path; active grid verified against config. Segment IDs stay runtime inputs.",
        expected_kernel_name="splash_mha_fwd_segmented_residuals",
        extra_rmsnorm=False, projections=False, collectives=False, backward=False,
    )
