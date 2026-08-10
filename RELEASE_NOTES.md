This is the initial public release of LGCR, a VapourSynth 4 plugin for
luma-guided chroma reconstruction. LGCR is designed to reconstruct subsampled
chroma while using full-resolution luma structure as a guide.

## Highlights

- `lgcr.Recon` supports same-size chroma reconstruction and combined resize
  and reconstruction for planar YUV 4:2:0, 4:2:2, and 4:4:4 clips.
- Three reconstruction modes are available: guided reconstruction (`algo=2`),
  per-pixel candidate selection (`algo=4`), and constrained luma-detail
  transfer (`algo=6`).
- Chroma siting is read from `_ChromaLocation` when available and can be
  overridden with `loc="left"` or `loc="center"`.
- `lgcr.TRecon` adds motion-compensated temporal reconstruction for
  constant-size clips with a known frame count.
- `lgcr.Sharpen` provides edge-aware sharpening while preserving the input
  dimensions, subsampling, sample type, and bit depth.
- 8-16-bit integer and 32-bit float planar YUV formats are supported.
- Pre-built AVX2/FMA binaries are provided for Linux x86-64 and Windows
  x86-64. A scalar reference build is available from source.

## Quick Start

Load the downloaded plugin and run the default spatial reconstruction:

```python
import vapoursynth as vs

core = vs.core
core.std.LoadPlugin("/absolute/path/to/liblgcr.so")  # Use lgcr.dll on Windows.

src = core.ffms2.Source("input.mkv")
out = core.lgcr.Recon(src)
out.set_output()
```

`Recon` and `TRecon` output YUV 4:4:4. Convert the result back to 4:2:0 only
when required by the delivery format. See the README for the complete API and
the parameter tuning guide.

## Release Assets

- `lgcr-<version>-linux-x86_64.tar.gz`: `liblgcr.so`, README, and MIT license.
- `lgcr-<version>-windows-x86_64.zip`: `lgcr.dll`, README, and MIT license.

The packaged binaries require an x86-64 CPU with AVX2 and FMA support and a
VapourSynth installation using API v4.

## Performance Note

On the current AMD Ryzen 9 5950X development host, the default spatial path
(1920x1080 YUV 4:2:0, Lanczos3, `algo=2`, `sparse=1`) measured approximately
**11.0 FPS with one VapourSynth thread**. This is the median of 30 frames
(90.81 ms/frame) with production profiling disabled. Performance depends on
the source, settings, CPU, compiler, and host load; treat this as a reference,
not a guarantee.

## Known Limitations

- The VapourSynth filters currently select the CPU backend. CUDA buffer,
  stream, dispatch, timing, and fallback infrastructure is included, but the
  compute kernels are still being connected.
- Inputs must be constant-size planar YUV. `Recon` supports 4:2:0, 4:2:2, and
  4:4:4; `TRecon` additionally requires a known frame count.
- Chroma reconstruction is content- and degradation-dependent. Test the
  defaults on representative material and compare with `strength=0` before
  tuning. This release does not claim a universal improvement over every
  conventional resampler or external method.

Linux release builds run the full correctness, regression, backend-matrix,
evaluation-protocol, and paper-consistency checks before publication. The
project is available under the MIT License.
