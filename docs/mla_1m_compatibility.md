# 1M baseline compatibility integration

This experiment imports the eight added files of primatrix/tokamax#28 at
`a21571c5d5b49b87899562d7fc553ab26fb303a4` onto the running 1M dependency
`cd79b342c3d408287e1dd76f677d227dca17640b`. Existing native kernels are unchanged.

The native Splash file differs from the PR's pin only by the existing
`fwd_kv_unroll` field and the two forward loop unroll arguments. Its backward
source and Ref argument ordering are identical. The new fused backward's source
guard is bound to this exact old native file, SHA256
`13ac6f78550a4e36f5f2d2cf32a05e94a892a8c42e356be7e3de72ef17bfa5ff`.
The CPU geometry, saved-residual, ownership and source-guard tests must be rerun;
changing the expected hash alone is not validation.

The new fused forward also honors this existing optional unroll setting.
Its default remains the PR's unrolled loop. The 1M recipe keeps the old rolled
setting. Arithmetic order is unchanged, but CPU AOT and a device run are still
separate gates. The distribution uses the local version `0.0.12+mla1940`, so
strict runtime ABI checks cannot confuse it with the baseline package.

The1M trace exposed quadratic host mask processing in the imported factory.
For a plain square zero-offset causal mask with unsharded metadata, the
integration derives each tile's empty/partial/full class from its endpoint
inequalities and reuses the native schedule generator. Every returned field
and dtype matches native processing across40 static/dynamic, forward/backward,
tile-ratio and dtype cases; other masks fall back. A complete1M/CP16/one-local-head
forward/backward trace with the peer barrier passed in the same42-test run in
2.07s. The old trace was still scanning the dense mask after21 minutes.
This is a host metadata construction improvement, not a device timing claim.
