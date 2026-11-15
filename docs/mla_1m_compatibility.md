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
