# Reduced-precision complex FFT proof

`reduced_complex_fft.py` exercises MLX's generic Metal C2C FFT with native
`complex_t<half>` and `complex_t<bfloat16_t>` storage without adding a public
dtype or changing production dispatch. Python supplies raw complex values as
interleaved scalar arrays with shape `[..., 2]`; the Metal boundary reinterprets
that packed layout as `complex_t<T>`.

The four representative cases force each C2C planner:

| Case | Size | Planner |
| --- | ---: | --- |
| Stockham | 256 | Mixed-radix shared-memory FFT |
| Rader | 17 | Prime-length Rader FFT |
| Bluestein | 47 | Prime length whose `N - 1` cannot use Rader |
| Four-step | 8192 | Two-pass power-of-two FFT |

For each planner and lane type, the script measures forward error against
NumPy, forward/inverse round-trip error, latency, raw-storage I/O throughput, and
speed relative to float32 complex storage. References are calculated from the
input after quantization to the tested lane type.

Timing samples amortize synchronization over 50 independent dispatches by
default. Use `--dispatches-per-sample` to change that value.

The optional static audit compiles focused FFT metallibs with the complete C2C
kernel matrix for each reduced lane. It reports binary-size and exported-symbol
growth without changing `fft.metal` or the production metallib.

Run the complete proof from the repository root:

```shell
python benchmarks/python/reduced_complex_fft.py \
  --static-audit \
  --full-metallib-build build \
  --json reduced_complex_fft.json
```

Use `--no-benchmark` for correctness-only validation. `--cases` and `--lanes`
accept comma-separated subsets for focused development runs.

`--full-metallib-build` is optional. When supplied, the audit replaces the FFT
AIR module in that CMake build and relinks all companion modules, reporting the
size delta for the complete `mlx.metallib` as well as the focused FFT library.

This is an engineering proof, not a proposed user-facing API. Public dtype
semantics, promotion, dispatch coverage, accuracy policy, and static kernel
selection remain separate design decisions.
