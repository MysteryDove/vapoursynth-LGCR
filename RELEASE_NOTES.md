# LGCR v2.1.0

LGCR v2.1.0 improves CPU reconstruction memory use and throughput while
preserving the plugin API, parameters, output formats, and quality semantics.

## Highlights

- Same-size Recon now shares the source luma plane with the output frame.
- Float Recon uses borrowed input/output views and avoids full-resolution
  conversion copies on the main path.
- Frame workspaces use bounded size-class scratch reuse.
- Algo6 streams same-size detail reconstruction through bounded row rings.
- Mutual-gate Sobel gradients use a four-pixel-halo row ring instead of six
  full-resolution gradient planes.
- Integer output remains bit-exact and float output remains within the existing
  numerical tolerance.

## 1080p Benchmark

Measured on an AMD Ryzen 9 5950X with a 1920x1080 YUV 4:2:0 Lanczos3 source,
`sparse=1`, production profiling disabled, one 1-second warm-up and a 5-second
throughput window. CPU affinity was pinned to `0`, `0-7`, and `0-15`.

| Algorithm | 1 thread | 8 threads | 16 threads |
|---|---:|---:|---:|
| Algo2 | 25.20 FPS | 172.20 FPS | 260.60 FPS |
| Algo4 | 18.20 FPS | 127.00 FPS | 198.60 FPS |
| Algo6 | 20.20 FPS | 136.00 FPS | 184.00 FPS |

These figures are host reference measurements; actual throughput depends on
source content, VapourSynth scheduling, CPU frequency, and compiler settings.

## Validation

- `make check`
- `make asan-check`
- 592-case numerical comparison: float maximum error `2.98e-08`, integers
  bit-exact
- AVX2/scalar, backend matrix, concurrency, battery, and evaluation consistency
  checks

## Assets

- `lgcr-v2.1.0-linux-x86_64.tar.gz`
- `lgcr-v2.1.0-windows-x86_64.zip`

Both archives contain the plugin binary, README, and MIT license.
