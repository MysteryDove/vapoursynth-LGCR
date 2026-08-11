# LGCR

LGCR is a VapourSynth 4 plugin for luma-guided chroma reconstruction. It
provides same-size chroma upsampling, combined resize and reconstruction,
optional temporal reconstruction, and edge-aware sharpening for planar YUV
clips and it works well with chroma-aliasing and chroma-bleeding in animation.

This repository is the source distribution. Release assets are pre-built for
Linux x86-64 and Windows x86-64; choose the asset matching the host CPU and
keep VapourSynth's `_ChromaLocation` frame property when possible.

## Visual Example

The comparison below shows the same crop from left to right: **Original**,
**Plain** reconstruction (`strength=0`), and **LGCR** reconstruction.

![LGCR comparison: Original | Plain | LGCR](compare_coat.png)

## Requirements

- VapourSynth R55 or newer with API v4 headers
- A C++17 compiler with `std::cyl_bessel_j` support
- GNU Make
- AVX2 and FMA for the default build

The scalar build does not require AVX2 or FMA.
Tests and evaluation additionally require Python 3, NumPy, and Pillow in the
VapourSynth Python environment.

CUDA is optional. The first CUDA-enabled release provides the device-buffer,
stream, stage-dispatch, timing, and CPU-fallback framework; compute kernels are
being connected stage by stage, so the VapourSynth filter still selects the CPU
backend by default.

## Build

The Makefile defaults to the header path under `~/vapoursynth`. Override
`VSINCLUDE` when the headers are elsewhere:

```sh
make VSINCLUDE=/path/to/vapoursynth/include
```

This creates `liblgcr.so`. To build the scalar reference plugin as well:

```sh
make liblgcr_scalar.so VSINCLUDE=/path/to/vapoursynth/include
```

To compile the optional CUDA framework against a CUDA toolkit:

```sh
make LGCR_ENABLE_CUDA=1 CUDA_ROOT=/usr/local/cuda
make cuda-framework-check CUDA_ROOT=/usr/local/cuda
```

With `LGCR_ENABLE_CUDA=0` (the default), no CUDA headers, runtime, or device are
required. CUDA availability does not change the public VapourSynth API.

To run the test suite, set `PYTHON` if VapourSynth is not installed in the
default local environment:

```sh
make check PYTHON=/path/to/vapoursynth/bin/python3
```

For a quick repeatable public-boundary performance baseline, run:

```sh
make benchmark PYTHON=/path/to/vapoursynth/bin/python3
```

The benchmark emits JSONL records per frame and per-case summaries. Records
include wall time, internal stage times, checksums, process memory, stage pixel
and tap counts, sparse-mask hit rate, and optional AVX2/scalar maximum error.
Every measured request uses a distinct frame number to avoid frame-cache hits.

The full 1080p/4K, 420/422/444, kernel, algorithm, sparse, and gate matrix is:

```sh
make benchmark BENCHMARK_ARGS="--preset full --iterations 3 --output baseline.jsonl"
```

Compare a later run with `--baseline baseline.jsonl`; summary records include
the resulting speedup. The checked-in pre-framework smoke reference can be
used on the original Ryzen 9 5950X host with:

```sh
make benchmark BENCHMARK_ARGS="--preset smoke --iterations 5 --baseline test/baselines/lgcr-smoke-pre-framework.jsonl --output current.jsonl"
```

Performance baselines are host- and toolchain-specific; use the checked-in
file as an informational reference or generate a fresh local baseline before
tuning. Add `--compare-scalar` after building `liblgcr_scalar.so` to record
backend error.

## Load The Plugin

Load the shared library explicitly in a VapourSynth script:

```python
import vapoursynth as vs

core = vs.core
core.std.LoadPlugin("/absolute/path/to/liblgcr.so")
```

Alternatively, install `liblgcr.so` in a VapourSynth autoload directory. The
functions are then available as `core.lgcr.Recon`, `core.lgcr.Downsample`,
`core.lgcr.TRecon`, and `core.lgcr.Sharpen`.

## Function Reference

### `lgcr.Recon`

```python
lgcr.Recon(
    clip,
    width=None,
    height=None,
    kernel="lanczos",
    taps=3,
    algo=2,
    b=0.0,
    c=0.6,
    strength=0.8,
    sigma=0.01,
    sratio=0.15,
    sdb=3.0,
    stretch=1.0,
    gsigma=2.5,
    rescue=1.0,
    ridge=1,
    cedge=0,
    ar=0.0,
    reg=0.005,
    loc=None,
    sparse=1,
    bp=0.0,
    ms=1.0,
    qgate=1.0,
    bm=False,
)
```

`Recon` accepts constant-size planar YUV 4:2:0, 4:2:2, or 4:4:4 input in
8-16-bit integer or 32-bit float format. It returns planar YUV 4:4:4 with the
same sample type and bit depth.

#### Same-Size 4:2:0 To 4:4:4

```python
src = core.ffms2.Source("input.mkv")
out = core.lgcr.Recon(src)
out.set_output()
```

The default is `algo=2` with a three-lobe Lanczos base kernel.

#### Resize And Reconstruct

```python
out = core.lgcr.Recon(
    src,
    width=3840,
    height=2160,
    kernel="jinc",
    taps=3,
    algo=2,
)
```

Both luma and chroma are resized. The output remains YUV 4:4:4.

#### Select A Reconstruction Mode

```python
guided = core.lgcr.Recon(src, algo=2)
selector = core.lgcr.Recon(src, algo=4)
detail_transfer = core.lgcr.Recon(src, algo=6, kernel="lanczos", taps=4)
radial_detail_transfer = core.lgcr.Recon(src, algo=6, kernel="jinc", taps=3)
plain = core.lgcr.Recon(src, strength=0.0)
```

- `algo=2` is the default guided reconstruction mode.
- `algo=4` selects among guided, local-regression, and plain candidates.
- `algo=6` applies constrained luma-detail transfer to the base reconstruction.
- `strength=0.0` disables guidance and is useful for A/B comparisons.

#### Optional Collaborative Block Refinement

```python
block_refined = core.lgcr.Recon(src, bm=True)
```

`bm=True` enables an experimental BM3D-inspired basic stage after chroma
reconstruction. It groups similar 8x8 patches on an 8-pixel anchor grid using
luma and the reconstructed U/V planes, applies separable 3D Haar hard
thresholding, and aggregates the filtered patches. The stage runs before
same-size 4:2:0 back-projection, so `bp` can still enforce source-sample
consistency.

For noisy sources, prefer a dedicated spatial or temporal denoiser before
LGCR when one is already available in the processing chain. This cleans both
the luma guide and the source chroma; leave `bm=False` unless visible chroma
noise remains. When upstream denoising is unavailable, `bm=True` provides a
self-contained alternative for residual chroma noise:

```python
# Preferred when a dedicated denoiser is already part of the script.
denoised = your_denoiser(src)
reconstructed = core.lgcr.Recon(denoised)  # bm=False by default

# Self-contained alternative when no upstream denoiser is available.
reconstructed_with_bm = core.lgcr.Recon(src, bm=True)
```

The built-in BM stage filters only the reconstructed U/V planes. It does not
denoise luma, repair compression artifacts before luma guidance, or replace a
dedicated temporal denoiser. Avoid stacking it automatically after strong
upstream denoising: the remaining gain may be small, while unmatched soft
chroma detail can be softened. The paired synthetic study found the clearest
benefit on noisy input; see the BM refinement results linked below.

The implementation is independent and includes no source code from
[VapourSynth-BM3DCUDA](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA),
which is GPL-2.0. Its high-performance block-search and aggregation design, and
the [original BM3D method of Dabov et al.](https://doi.org/10.1109/TIP.2007.901238),
informed this optional path. The stage is currently CPU-only, uses additional
full-resolution float buffers, and is not part of the frozen paper evaluation.
AVX2 builds transform each four-patch U/V group across SIMD lanes; scalar builds
use the same algorithm as a portability and correctness reference. On the
current Ryzen 9 5950X development host, the 1920x1080 single-thread reference
case measured approximately 255 ms/frame with `bm=True` versus 86 ms/frame
with `bm=False`. Performance depends on the source and host; treat these as
relative reference figures rather than a guarantee.

#### Chroma Siting

When `loc` is omitted, `Recon` reads the frame's `_ChromaLocation` property.
Preserve this property from the source whenever possible because it can encode
both horizontal and vertical siting. If neither metadata nor `loc` is present,
the fallback is horizontally left-sited and vertically centered chroma.

For sources without metadata, `loc` can override horizontally left-sited or
center-sited chroma:

```python
left = core.lgcr.Recon(src, loc="left")
center = core.lgcr.Recon(src, loc="center")
```

The explicit `loc` override represents vertically centered chroma. For top- or
bottom-sited material, set the correct `_ChromaLocation` frame property and
leave `loc` unset.

#### Recon Parameters

| Parameter | Default | Accepted values / purpose |
|---|---:|---|
| `width`, `height` | input size | Positive output dimensions |
| `kernel` | `"lanczos"` | `"bilinear"`, `"bicubic"`, `"lanczos"`, `"spline16"`, `"spline36"`, or `"jinc"` |
| `taps` | `3` | Lanczos/Jinc support, `1..64` |
| `b`, `c` | `0.0`, `0.6` | Bicubic parameters |
| `algo` | `2` | `2`, `4`, or `6` |
| `strength` | `0.8` | Guidance strength, `0..1`; `0` selects plain reconstruction |
| `loc` | frame property | `"left"` or `"center"` horizontal siting override |
| `sparse` | `1` | Skip guided work away from detected luma structure |
| `ar` | `0.0` | Local chroma-hull margin; a negative value disables clamping |
| `sigma` | `0.01` | Positive luma-similarity floor |
| `sratio` | `0.15` | Positive adaptive-similarity ratio |
| `sdb` | `3.0` | Positive ramp-similarity multiplier |
| `stretch` | `1.0` | Non-negative along-edge support stretch |
| `gsigma` | `2.5` | Positive guided-support width |
| `rescue` | `1.0` | Phase-rescue scale, `0..1` |
| `ridge` | `1` | Thin-line protection toggle |
| `ms` | `1.0` | Mutual-structure gate strength, `0..1` |
| `qgate` | `1.0` | Algo 6 affine-confidence gate strength, `0..1` |
| `reg` | `0.005` | Positive regularization for algos 4 and 6 |
| `cedge` | `0` | Experimental chroma-transition gate toggle |
| `bp` | `0.0` | Same-size 4:2:0 back-projection gain, `0..1` |
| `bm` | `False` | Enable experimental BM3D-style collaborative chroma refinement |

### `lgcr.Downsample`

```python
lgcr.Downsample(
    clip,
    quality=0,
    kernel="spline36",
    strength=1.0,
    loc="left",
)
```

`Downsample` accepts only constant-size, even-dimension planar YUV 4:4:4 in
8-16-bit integer or 32-bit float format. It preserves the frame width, height,
sample type, and bit depth while returning YUV 4:2:0. The Y plane is attached
to the output frame through VapourSynth's plane-sharing API and is bit-exact;
only U and V are computed.

This is a standalone 4:4:4 to 4:2:0 path. It does not call `Recon`, does not use
`algo=6`, and does not allocate a full-frame float guide or tensor map. The
plain separable result is generated through a short row ring. Guided weights
are then evaluated directly on the output grid from local Y/U/V samples, with
shared direction classification for both chroma planes. Flat luma, isolated
chroma transitions, direction mismatches, and low-confidence structure fall
back to the plain result.

| `quality` | Local support | Direction/candidate tradeoff |
|---:|---:|---|
| `0` | up to 5x5 | Fast; isotropic, horizontal, vertical, and two diagonal classes with a hard direction choice |
| `1` | up to 7x7 | Balanced; eight directions plus isotropic, softly blending the nearest two directions |
| `2` | up to 9x9 | High; keeps short rings of directional/anisotropic candidates and selects them with a luma-edge-weighted fixed bilinear 2x loopback score |

Quality 2 is a local approximation inspired by the Wang-style loopback idea,
not a reproduction of a published algorithm. It stores only the candidate rows
needed by the current score window and creates no internal thread pool.

`strength=0` selects the same plain baseline in every quality mode. Values up
to `1` continuously blend towards the gated guided result. The correction is a
convex move from the baseline towards a positive-weight local chroma estimate,
so it cannot increase the baseline's local-hull overshoot.

The output location is explicit and is written to `_ChromaLocation` using the
corresponding H.273 value:

```python
left = core.lgcr.Downsample(src444, loc="left")
center = core.lgcr.Downsample(src444, loc="center")
top_left = core.lgcr.Downsample(src444, loc="topleft")
top = core.lgcr.Downsample(src444, loc="top")
bottom_left = core.lgcr.Downsample(src444, loc="bottomleft")
bottom = core.lgcr.Downsample(src444, loc="bottom")
```

| Parameter | Default | Accepted values / purpose |
|---|---:|---|
| `quality` | `0` | `0` fast, `1` balanced, or `2` high |
| `kernel` | `"spline36"` | `"spline36"`, `"lanczos3"`, or `"binomial"` plain baseline |
| `strength` | `1.0` | Guided blend, `0..1`; `0` is the plain baseline |
| `loc` | `"left"` | `"left"`, `"center"`, `"topleft"`, `"top"`, `"bottomleft"`, or `"bottom"` |

Spline36 and Lanczos3 use the shared polyphase weight generator. `binomial`
uses fixed normalized phases: a five-tap `[1, 4, 6, 4, 1] / 16` filter at
integer sample positions and a four-tap `[1, 3, 3, 1] / 8` filter at
half-integer positions.

### `lgcr.Sharpen`

```python
lgcr.Sharpen(
    clip,
    alpha=0.3,
    sigma=0.01,
    sratio=0.15,
    gspatial=1.2,
    ar=0.0,
)
```

`Sharpen` accepts planar YUV input and preserves its dimensions, subsampling,
sample type, and bit depth.

```python
reconstructed = core.lgcr.Recon(src)
out = core.lgcr.Sharpen(reconstructed, alpha=0.3)
```

It can also be used directly on subsampled YUV:

```python
out = core.lgcr.Sharpen(src, alpha=0.4, ar=0.0)
```

#### Sharpen Parameters

| Parameter | Default | Accepted values / purpose |
|---|---:|---|
| `alpha` | `0.3` | Sharpening amount |
| `sigma` | `0.01` | Positive similarity floor |
| `sratio` | `0.15` | Positive adaptive-similarity ratio |
| `gspatial` | `1.2` | Positive spatial support width |
| `ar` | `0.0` | Local sample-hull margin; a negative value disables clamping |

### `lgcr.TRecon`

```python
lgcr.TRecon(
    clip,
    strength=0.8,
    sigma=0.01,
    sratio=0.15,
    sdb=3.0,
    gsigma=2.5,
    stretch=1.0,
    ar=0.0,
    ridge=1,
    sparse=0,
    ms=1.0,
    trad=1,
    tsearch=6,
    tsad=0.02,
    bm=False,
)
```

`TRecon` performs same-size temporal reconstruction and returns YUV 4:4:4. It
requires a constant frame size and frame count. The base spatial mode is fixed
to algo 2 with Lanczos3.

```python
temporal = core.lgcr.TRecon(src, trad=1)
temporal_wide = core.lgcr.TRecon(src, trad=2, tsearch=8)
temporal_block_refined = core.lgcr.TRecon(src, trad=1, bm=True)
```

Use `Recon` for still images, single-frame clips, or when temporal motion
matching is not desired.

#### TRecon Parameters

| Parameter | Default | Accepted values / purpose |
|---|---:|---|
| `trad` | `1` | Temporal radius, `0..8` frames on each side |
| `tsearch` | `6` | Integer motion-search radius, `0..64` luma pixels |
| `tsad` | `0.02` | Positive motion-match confidence scale |
| `strength` | `0.8` | Guidance strength, `0..1` |
| `sigma` | `0.01` | Positive luma-similarity floor |
| `sratio` | `0.15` | Positive adaptive-similarity ratio |
| `sdb` | `3.0` | Positive ramp-similarity multiplier |
| `gsigma` | `2.5` | Positive guided-support width |
| `stretch` | `1.0` | Non-negative along-edge support stretch |
| `ar` | `0.0` | Local chroma-hull margin; a negative value disables clamping |
| `ridge` | `1` | Thin-line protection toggle |
| `sparse` | `0` | Sparse guided-work toggle |
| `ms` | `1.0` | Mutual-structure gate strength, `0..1` |
| `bm` | `False` | Enable post-reconstruction BM3D-style collaborative chroma refinement |

## Parameter Tuning Guide

Start with the defaults and change one parameter at a time. Compare against
`strength=0` (plain reconstruction) on a short representative clip. These
are practical directions, not quality guarantees; extreme values can trade
ringing, softness, and false colour.

### Recon

| If you see / need | Try | What to expect |
|---|---|---|
| More detail is still missing at chroma edges | `strength` up (towards `1`) | More luma guidance; can copy luma noise or create colour halos |
| False colour or luma texture is leaking into chroma | Lower `strength`; lower `sigma`/`sratio` | Tighter luma matching and a plainer result; too low can reject useful neighbours |
| Smooth ramps fragment or remain aliased | Raise `sdb` slightly | Broader matching on ramps; too high can smear transitions |
| Diagonals or mixed edge types need a better choice | `algo=4` | Selects among guided, regression, and plain candidates; slower than `algo=2` |
| You want crisp detail transfer after reconstruction | `algo=6`; adjust `qgate` | Transfers detail only when its affine fit is credible; lower `qgate` is more permissive |
| Thin lines break or disappear | Keep `ridge=1`; raise `rescue` | Protects ridge/phase-zero structure; may retain more noise |
| Wide chroma transitions look over-sharpened | Try `cedge=1` | Experimental transition fade; validate on the target material |
| Edge support is too isotropic or too broad along an edge | Adjust `stretch` (0 is isotropic) and `gsigma` | Higher values extend/soften support and can reduce locality |
| Flat areas are slow with little visible change | Keep `sparse=1` | Skips guidance away from luma structure; set `0` for debugging/A-B parity |
| Output rings or overshoots the local chroma range | Lower `ar` towards `0` | Tightens the local hull clamp; higher values allow more overshoot and negative disables clamping |
| Regression modes are unstable/noisy | Raise `reg` | Stronger regularization for `algo=4/6`, at the cost of detail |
| 4:2:0 average is not preserved after same-size reconstruction | Raise `bp` from `0` towards `1` | Adds data-consistency back-projection; too high can flatten edges |
| Chroma edges are shifted left/right | Set `loc="left"` or `loc="center"`, or fix `_ChromaLocation` | Correct siting before tuning quality parameters; `loc` only controls horizontal siting |
| Resizing changes the character of the result | Try `kernel` and `taps` | `bilinear` is soft, `bicubic` is tunable with `b/c`, higher Lanczos/Jinc taps are sharper and slower |
| The source contains luma and chroma noise | Denoise before `Recon`; keep `bm=False` initially | Gives LGCR a cleaner guide and lets a dedicated denoiser handle both luma and chroma |
| Repeated flat or textured regions retain chroma noise after reconstruction | Try `bm=True` | Adds self-contained non-local chroma filtering; costs substantial CPU time and memory, and may soften unmatched detail |

`b` and `c` only affect `kernel="bicubic"`: increase `b` for a smoother,
less ringing response; increase `c` for a sharper response. Keep both near
their conventional range and check overshoot. `ms` reduces guidance when
luma/chroma structure disagrees; lower it when textured chroma is being
suppressed, and raise it when false structure is appearing. `sigma` is the
absolute similarity floor, while `sratio` scales with the local luma range.
Higher values accept larger luma differences; lower values reject them.
With `bm=True`, `sigma` also seeds the block-match acceptance and transform
threshold; there is intentionally no second BM-specific strength control yet.
If an upstream denoiser already removed the visible chroma noise, compare
against `bm=False` before retaining the additional BM pass.

### Sharpen

| If you see / need | Try | What to expect |
|---|---|---|
| Reconstructed chroma is too soft | Raise `alpha` in small steps | More edge contrast; ringing/noise also increases |
| Sharpening catches noise or texture | Lower `alpha`; lower `sigma`/`sratio` | Tighter luma similarity reduces unrelated high-pass detail |
| The effect is too local or too broad | Adjust `gspatial` | Larger values spread the spatial support and soften the gate |
| Overshoot appears at colour boundaries | Lower `ar` towards `0` | Tightens the local hull clamp; negative disables clamping |

### TRecon

| If you see / need | Try | What to expect |
|---|---|---|
| Static noise remains | Raise `trad` (up to `8`) | Uses more neighbouring frames; costs memory/time |
| Motion is missed | Raise `tsearch` (up to `64`) | Finds larger integer motion; costs time and can match the wrong block |
| Valid temporal matches are being rejected | Raise `tsad` slightly | Makes the confidence gate more permissive; too high can ghost |
| Motion matches are unstable or ghost | Lower `tsad`; lower `trad` | Rejects weaker matches / uses fewer frames |
| Temporal result is too guided or too soft | Adjust `strength`, `sigma`, `sratio`, `sdb`, `gsigma`, `stretch`, `ridge`, `ar`, and `ms` as in `Recon` | TRecon uses the fixed algo-2 Lanczos3 spatial base |
| Flat areas need temporal averaging | Keep `sparse=0` | `sparse=1` saves guided work but skips corrections away from luma structure |
| Motion compensation is stable but residual chroma noise remains | Try `bm=True` | Applies collaborative spatial refinement after temporal reconstruction; it does not replace motion matching |
| A single-frame or variable-length clip is used | Use `Recon` instead | TRecon requires constant dimensions and a known frame count |

For a first pass, use `Recon(src)` for stills and `TRecon(src, trad=1)` for
stable video. Tune siting and the base kernel before changing algorithmic
gates. For noisy video, run an existing temporal denoiser before `TRecon` when
possible; use `bm=True` as the built-in residual-chroma alternative. Always
inspect animation line art and natural-texture shots separately.

## Output And Encoding

`Recon` and `TRecon` return 4:4:4. Downsample only when required by the delivery
format:

```python
reconstructed = core.lgcr.Recon(src)
delivery = core.lgcr.Downsample(
    reconstructed,
    quality=1,
    kernel="spline36",
    loc="left",
)
delivery.set_output()
```

Example command-line pipeline:

```sh
vspipe --y4m script.vpy - | ffmpeg -i - -c:v libx265 -crf 16 output.mkv
```

## Paper And Evaluation

- [Working paper](paper/paper.md)
- [Publication plan and evidence ledger](paper/PLAN.md)
- [Evaluation protocol](evaluation/protocol.md)
- [Generated results](evaluation/results/test.md)
- [BM refinement study protocol](evaluation/bm_study_protocol.md)
- [BM refinement study results](evaluation/results/bm_study.md)

Regenerate the synthetic evaluation with the VapourSynth-enabled Python
interpreter:

```sh
make eval-results PYTHON=/path/to/vapoursynth/bin/python3
```

## Releasing

Maintainers can run the `release` workflow from the GitHub Actions page. Select
the source ref, enter a new `vMAJOR.MINOR.PATCH` tag, and optionally mark it as
a pre-release. The workflow runs the Linux checks, builds Linux and Windows
assets, and creates the tag and GitHub Release only after both builds succeed.

## License

LGCR is available under the [MIT License](LICENSE).
