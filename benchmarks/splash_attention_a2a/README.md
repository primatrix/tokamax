# CP8 Splash/A2A benchmark

The benchmark keeps the input construction in `inputs.py`.  `run_baseline.py`
profiles native Ulysses pre-A2A + Splash512 + post-A2A.  `run_optimized.py`
profiles the fused head-overlap implementation in
`tokamax._src.ops.experimental.tpu.splash_attention.splash_attention_a2a_kernel`.

The two extraction scripts operate on the same XPlane layout:

```bash
python profile_baseline.py --trace-dir <baseline>/profile/trace \
  --out <baseline>/baseline-device-times.json
python profile_optimized.py --trace-dir <optimized>/profile/trace \
  --out <optimized>/optimized-device-times.json
```

Only `run_baseline.py` is used to establish the frozen baseline.  Later
iterations run `run_optimized.py` and write each result under a new
`/gcs/kkx/tokamax-ling3-v0.2.2/iterations/<iteration>/` directory.
