# From 4:2:0 Sampling to Color Bleed: A Controlled Study of Chroma Boundary Reconstruction for Animation-Style Content

**MysteryDove**  
[github.com/MysteryDove](https://github.com/MysteryDove)

Working paper, version 0.1, 2026-08-08

> Status: the controlled synthetic study is reproducible, but the licensed
> animation/natural-image corpus is not yet populated. This draft does not
> support a claim about prevalence or subjective benefit on released animation.

## Abstract

Chroma subsampling can turn a sharp color boundary into a displaced, widened,
stair-stepped, or ringing reconstruction. Animation-style imagery is a
plausible stress domain because it often combines saturated flat regions with
drawn high-contrast boundaries, but prior work already includes decoder-side
color-bleeding removal and cel-like artificial tests. The remaining question
is therefore not whether color bleeding exists, but how reconstruction methods
behave when the original downsampling kernel and chroma siting are uncertain
and when luma and chroma structures disagree.

We provide a deterministic 4:2:0 forward model, an animation-style synthetic
scene generator, and operational measures for boundary-profile error,
transition spread, tangent aliasing, systematic phase displacement, and
ringing. We compare four signal-only resamplers, five analytic guided methods,
plain Jinc, and three variants of Luma-Guided Chroma Reconstruction (LGCR).
Methods and external comparators are selected on 12 development scenes. The
held-out benchmark contains 32 scenes, four degradation kernels, and two
chroma sitings; uncertainty is estimated by paired scene bootstrap.

The preregistered LGCR variant, algo6, improves edge-band MAE over plain Jinc by
0.002129 (95% CI 0.000959 to 0.003326), but it does not beat the strongest
external comparator. Wada et al.'s extended joint bilateral filter (EJBF)
achieves 0.021545 edge MAE versus 0.023654 for algo6; the paired algo6-minus-EJBF
difference is +0.002109 (95% CI +0.000381 to +0.003706). The ranking reverses
by degradation: algo6 wins for box and triangle downsampling and loses for
bicubic and held-out Lanczos. A siting intervention nearly doubles algo6's
absolute phase error from 0.350 to 0.692 pixels. These results support an open,
mechanism-oriented benchmark and a conditional reconstruction tradeoff, not a
state-of-the-art claim. Real-animation prevalence, codec quantization, and
subjective benefit remain open evaluation requirements.

Keywords: chroma upsampling, color bleeding, chroma aliasing, animation,
4:2:0, chroma siting, guided reconstruction, reproducibility

## 1. Introduction

Most consumer video stores luma at full spatial resolution and chroma on a
coarser grid. For 4:2:0, each chroma plane has one quarter as many spatial
samples as luma. Sampling is not intrinsically defective: when chroma is
appropriately band-limited and its grid is reconstructed with the correct
phase, it is an efficient representation. The difficult case is a strong color
boundary whose energy extends beyond the chroma Nyquist limit. Downsampling
then discards boundary location and shape information, and a decoder must
choose how to interpolate what is no longer observed.

The resulting artifacts are often grouped under the phrase "color bleeding."
That phrase hides several different visible outcomes. A wide positive kernel
can spread a transition; an insufficient reconstruction filter can leave a
staircase along diagonal boundaries; a siting mismatch can displace a boundary;
and a negative-lobed kernel can overshoot into a colored halo. These outcomes
share a sampling history but need not respond to the same repair.

Animation is a motivated, not yet empirically established, target domain.
Flat, saturated regions and drawn outlines plausibly create more strong
luma/chroma co-edges than natural photography. Wada et al. explicitly noted
cel animation as a motivating artificial-image case in 2015
([doi:10.3169/mta.3.95](https://doi.org/10.3169/mta.3.95)). That prior result is
important: neither decoder-side debleeding nor cel-like synthetic evaluation
is new. A real domain claim also requires source-level 4:4:4 animation, not a
screenshot that has already passed through 4:2:0.

### 1.1 Causal chain and artifact scope

The animation hypothesis begins with signal structure, not with a particular
filter. An ideal flat-color boundary is a chroma step and is therefore not
band-limited. When a dark drawn contour or a simultaneous luma transition lies
at the same position, full-resolution luma supplies side information about the
location of chroma detail that 4:2:0 no longer samples. Saturated color pairs
also produce a larger absolute Cb/Cr step for a given spatial error. These
properties motivate animation-style stress scenes, but whether they occur more
often in released animation than in photographs is an empirical question left
to the incomplete corpus study.

The controlled causal chain is:

1. **Source boundary.** A sharp full-resolution chroma transition contains
   energy above the half-rate chroma Nyquist limit. This is a difficult input,
   not yet an artifact.
2. **Prefilter and 4:2:0 sampling.** The encoder-side low-pass kernel trades
   alias suppression for transition spread, then retains one chroma sample per
   two-by-two luma sampling cell. Boundary values and subpixel location are
   no longer uniquely recoverable. Different unknown prefilters leave
   different edge profiles in the same nominal 4:2:0 format.
3. **Grid interpretation.** Chroma siting specifies where those retained
   samples lie relative to luma. Treating left-sited samples as centered, or
   the reverse, introduces a systematic phase error even when the interpolation
   kernel itself is unchanged. H.273 metadata can describe this grid, but a
   decoder may receive missing or incorrect metadata.
4. **Decoder reconstruction.** A broad positive response tends to preserve the
   prefiltered transition as visible spread; inadequate suppression of sampled
   high-frequency images appears as tangent stair-stepping on sloped edges; and
   negative lobes can create overshoot or a complementary-color halo. Thus
   bleeding, aliasing, phase displacement, and ringing are related outcomes of
   information loss and reconstruction choice, not interchangeable names for
   one scalar defect.

Later operations can add different damage: separate-channel sharpening can
create new color fringes, an incorrect interlaced chroma conversion can mix
fields, and composite-video cross-color has a different physical origin. Those
cases, as well as neural enhancement, are excluded here so that the experiment
isolates progressive 4:2:0 boundary reconstruction.

### 1.2 Prior work and the narrowed gap

Decoder-side color-bleeding postprocessing predates modern guided upsampling.
Coudoux et al. proposed DCT-aware postprocessing in 2004 and a 4:1:1 video
method in 2005
([doi:10.1109/TCSVT.2003.819179](https://doi.org/10.1109/TCSVT.2003.819179),
[doi:10.1109/TBC.2005.852243](https://doi.org/10.1109/TBC.2005.852243)).
Catorina et al., Punchihewa and colleagues, and Li et al. studied adaptive
removal, subsampling/quantization effects, or objective measurement of color
bleeding
([doi:10.1117/12.703061](https://doi.org/10.1117/12.703061),
[doi:10.1109/IVCNZ.2008.4762087](https://doi.org/10.1109/IVCNZ.2008.4762087),
[doi:10.1049/cp:20080420](https://doi.org/10.1049/cp:20080420),
[doi:10.1109/ICCSNT.2011.6182054](https://doi.org/10.1109/ICCSNT.2011.6182054)).
Wada et al.'s EJBF is especially close: it uses Y, Cb, and Cr jointly to reduce
both compression noise and 4:2:0 blur after decoding.

Joint bilateral upsampling and guided filtering established general
high-resolution-guide reconstruction
([Kopf et al., 2007](https://doi.org/10.1145/1276377.1276497);
[He et al., 2013](https://doi.org/10.1109/TPAMI.2012.213)). Anisotropic guided
filtering further adapts spatial support to local orientation
([Ochotorena and Yamashita, 2020](https://doi.org/10.1109/TIP.2019.2941326)).
Encoder-coupled chroma work uses luma when choosing both downsampling and
reconstruction. Examples include Korhonen's luma-assisted subsampling, Wang et
al.'s joint screen-content pipeline, Vermeir et al.'s guided screen-content
reconstruction, the methods of Chung et al., and Fu et al.'s HDR/WCG pipeline
([Korhonen, 2015](https://doi.org/10.1109/ICME.2015.7177387);
[Wang et al., 2016](https://doi.org/10.1109/TCSVT.2015.2461891);
[Vermeir et al., 2016](https://doi.org/10.1109/TCSVT.2015.2469118);
[Chung et al., 2017](https://doi.org/10.1109/TIP.2017.2749148);
[Chung et al., 2019](https://doi.org/10.1109/TIP.2018.2875340);
[Fu et al., 2019](https://doi.org/10.1109/ACCESS.2019.2911673)). GALOSH RAW
recently used a signed EWA-Jinc joint-bilateral chroma upsampler in a Bayer/CFA
pipeline ([Sato, 2026](https://doi.org/10.48550/arXiv.2607.03768)).

The defensible gap is narrower than "post-hoc chroma repair is empty." Existing
studies generally optimize a known codec or paired down/up process, target
compression noise as a whole, or report aggregate fidelity. Less attention is
given to decoder-only reconstruction under an unknown degradation family,
explicit H.273-like siting interventions, separate boundary mechanisms, and
guide-mismatch stress cases tailored to animation-style structure. The
detailed capability comparison is in
[related_work_matrix.md](related_work_matrix.md).

### 1.3 Research questions and contributions

This report asks:

1. Can bleeding/blur, tangent aliasing, phase displacement, and ringing be
   operationalized in a reproducible boundary-profile protocol?
2. How does a constrained luma-guided reconstruction compare with conventional
   resampling and existing guided methods after development-only selection?
3. How sensitive are conclusions to the unknown degradation kernel, chroma
   siting, and violations of the luma/chroma co-edge assumption?

The contributions are deliberately bounded:

1. A causal forward model and deterministic metric tests that distinguish four
   boundary manifestations without claiming they are statistically orthogonal.
2. A controlled animation-style benchmark with a development/test split,
   unknown-kernel holdout, explicit siting intervention, raw rows, and
   scene-clustered uncertainty.
3. A decoder-only LGCR implementation whose studied variant adds constrained
   luma detail to a signal-only base and gates transfer using chroma evidence.
4. A comparison that includes Wada EJBF and reports a negative primary result,
   degradation-specific rank reversals, and the largest failures.
5. A source-licensing and sampling contract that makes the missing real-domain
   evidence visible rather than silently substituting an unlicensed screenshot.

## 2. Methods

### 2.1 Scope and forward model

The main task is single-frame, training-free reconstruction from YUV 4:2:0 to
YUV 4:4:4. Let full-resolution chroma be $C$, the true separable degradation
kernel be $h_D$, chroma siting be $s=(s_x,s_y)$, and quantization/codec effects
be $Q$. The general observation model is

$$
C_{420}[m,n] = Q\left((h_D * C)[2m+s_x,2n+s_y]\right) + \eta.
$$

A decoder observes $C_{420}$ and full-resolution luma $Y$ and returns

$$
\hat C = R(C_{420},Y;\hat s,\theta).
$$

True siting $s$ and assumed siting $\hat s$ are separate factors. Pixel centers
use integer luma coordinates. In the implementation, a chroma sample is at
$2i+0.5+s_x$; `left` uses $s_x=-0.5$ and `center` uses $s_x=0$. H.273 metadata
is represented by VapourSynth's `_ChromaLocation` property.

The primary experiment sets $Q$ to the identity and $\eta=0$. It therefore
isolates subsampling and reconstruction. It does not establish robustness to
transform coding or chroma quantization.

### 2.2 Operational artifact measures

Each synthetic scene has two known chroma side colors
$c_0=(U_0,V_0)$ and $c_1=(U_1,V_1)$. A reconstruction is projected onto their
color line:

$$
\hat\alpha(x,y) =
\frac{(\hat U-U_0)(U_1-U_0)+(\hat V-V_0)(V_1-V_0)}
{(U_1-U_0)^2+(V_1-V_0)^2}.
$$

The ground-truth profile is $\alpha(x,y)$. For each valid image row, the
protocol finds the crossings of 0.1, 0.5, and 0.9 nearest the known boundary.
It then computes:

- **Phase displacement:** absolute mean error of the 0.5 crossing, in luma
  pixels. This measures a systematic shift, not rowwise jitter.
- **Tangent alias:** RMS first difference of rowwise 0.5-crossing errors,
  divided by $\sqrt{2}$. It responds to rapid staircase variation along the
  boundary.
- **Transition-spread delta:** predicted 0.1-to-0.9 width minus the true width.
  Positive values indicate a wider transition.
- **Bleed profile error:** mean $|\hat\alpha-\alpha|$ in a 15-pixel boundary
  profile. This is a holistic mixture error and can also respond to phase.
- **Ringing mass:** mean
  $\max(-\hat\alpha,0)+\max(\hat\alpha-1,0)$ in the same profile.

The primary endpoint is the mean absolute U/V error within four luma pixels of
the true chroma boundary. Full-frame MAE, PSNR, and MAE at least 10 pixels from
the boundary are secondary. Scenes with no chroma transition are excluded only
from crossing-profile measures. Unit tests independently inject a one-pixel
shift, alternating edge displacement, blur, and overshoot to verify metric
specificity.

These measures are operational descriptors, not an independent-source
decomposition. For example, a phase shift also increases profile error.

### 2.3 Synthetic scenes and degradation

Every scene is a deterministic 96 by 96 image with a straight, sinusoidal, or
chevron boundary and one of six evaluated conditions:

- `coedge`: luma and chroma transitions share position and width;
- `soft_chroma`: chroma width is 4 to 7 pixels under a sharper luma boundary;
- `misaligned`: chroma is displaced 0.75 to 1.25 pixels from luma;
- `isoluminant`: chroma changes while luma is constant across the boundary;
- `ridge`: a dark luma line is superimposed near the color transition;
- `luma_only`: chroma is constant across a luma transition.

Colors are drawn from a fixed saturated palette. Boundary phase, slope, width,
and curvature are seeded. Twelve development scenes use seeds 100 through 111.
Thirty-two held-out scenes use seeds 1000 through 1031, comprising 12 co-edge
scenes and four scenes for each other condition.

Full-resolution U and V are downsampled by normalized box, triangle, or
bicubic kernels on development. Held-out testing adds Lanczos as an unseen
kernel family. Development uses left siting. Testing crosses all four kernels
with left and center siting, producing 256 scene/degradation/siting conditions.
No test condition is used to select the primary LGCR variant.

### 2.4 Comparators

The benchmark includes:

- zimg bilinear, bicubic, Lanczos3, and Spline36;
- Gaussian joint bilateral upsampling (JBU);
- guided-filter upsampling;
- Wada et al.'s extended joint bilateral filter (EJBF);
- Korhonen's four-candidate luma-MSD rule;
- an H.273-grid adaptation of GALOSH RAW's signed EWA-Jinc JBU;
- plain Jinc3 and LGCR algorithms 2, 4, and 6.

The analytic methods are formula-level references, not bit-exact ports of full
author systems. JBU, guided filter, and the GALOSH-form range parameter are
tuned on development scenes. Wada EJBF uses its published normalized
parameters: spatial sigma 4, luma sigma 0.03, Cb/Cr sigma 0.10. Decoded chroma
is first bilinearly expanded, Eq. 10 is applied to Y, Cb, and Cr jointly, and
the spatial Gaussian is truncated at three sigma (radius 12).

The development set selects the lowest mean edge MAE among external methods
and separately among LGCR algorithms. Wada EJBF (`0.022346`) becomes the
external comparator; `lgcr_algo6` (`0.023963`) becomes the primary LGCR method.

This Wada comparison was added in protocol version 2 after the original LGCR
held-out run exposed a literature omission. To limit the resulting bias, the
published EJBF parameters and development-only comparator selection were
frozen before EJBF was executed on held-out scenes; the LGCR choice and all
LGCR parameters remained unchanged. This amendment is still weaker than a
single preregistration and is reported as a limitation.

### 2.5 Constrained luma detail transfer in LGCR algo6

The primary LGCR variant starts from a signal-only Jinc3 reconstruction $C_0$.
It does not replace low-frequency chroma with a luma regression. Instead it
adds a bounded estimate of detail missing under candidate downsamplers:

$$
\hat C = C_0 + \operatorname{clip}(g\,a\,d_{safe}).
$$

For candidate degradation kernels $D_k$ in {box, triangle, bicubic}, luma is
mapped to the chroma grid. In each 5 by 5 chroma window, separate U and V slopes
$a_k$ are estimated from local covariance. The applied slope is the median
over candidates. A joint affine credibility is

$$
q_k = \frac{\operatorname{cov}(Y_k,U)^2+
                 \operatorname{cov}(Y_k,V)^2}
{(\operatorname{var}(Y_k)+\epsilon)
 (\operatorname{var}(U)+\operatorname{var}(V)+\epsilon)}.
$$

At the evaluated default, transfer confidence is the mean $q_k$, multiplied by
$q_{min}/q_{max}$, a chroma-significance term, strength, and a mutual-structure
gate. The gate compares luma and chroma gradient profiles along the luma edge
normal on the chroma grid. It fades transfer when the chroma transition is much
wider or its gradient centroid is displaced.

The candidate luma maps are median-combined and reconstructed to the source
luma grid. The residual $Y-P(\operatorname{median}_k D_k(Y))$ is filtered by
`[1,2,1]/4` along each subsampled axis before output scaling. This rejects the
axial alternating mode at each dyadic stage; it is not claimed to span the
null space of an unknown encoder. A structure-tensor tangent filter favors
one-dimensional normal detail. Finally, each chroma correction is bounded by
half the local chroma range and the result is clamped to a local hull.

All primary LGCR runs use Jinc3, strength 0.8, dense evaluation, zero hull
margin, full affine credibility, and full mutual-structure gating. Algorithms
2 and 4 are reported as secondary variants. Algorithm 2 uses luma similarity
to modulate the signed interpolation kernel; algorithm 4 routes among that
path, a local guided regression, and the plain base.

### 2.6 Statistical analysis and reproducibility

The scene, not a pixel or degradation variant, is the independent unit. For a
method comparison, all conditions for each scene are first averaged; 4,000
bootstrap samples then resample scene identifiers with replacement. The report
uses percentile 95% confidence intervals and a fixed seed of 20260808. The
primary comparisons are algo6 versus Wada EJBF and algo6 versus plain Jinc.
Secondary metrics, degradation strata, condition strata, siting intervention,
and ablations are diagnostic and are not corrected for multiple comparisons.

The benchmark implementation, raw CSV files, generated Markdown reports, and
protocol are under `evaluation/`. The plugin is a C++17 VapourSynth plugin.
`make eval-results` regenerates all synthetic results; `make check` runs plugin,
scalar/AVX2, regression, battery, and metric tests.

## 3. Results

### 3.1 Held-out comparison

The held-out benchmark contains 32 scenes and 3,328 raw method-condition rows.
Table 1 gives scene-weighted means. Confidence intervals apply to edge MAE.

**Table 1. Held-out controlled synthetic benchmark. Lower is better; spread
delta is best near zero.**

| Method | Edge MAE [95% CI] | Bleed profile | Abs. phase px | Alias px | Spread delta px | Ringing |
|---|---:|---:|---:|---:|---:|---:|
| zimg Bilinear | 0.030343 [0.025018, 0.035095] | 0.041260 | 0.068325 | 0.092965 | 1.621131 | 0.002001 |
| zimg Bicubic | 0.027729 [0.022595, 0.032266] | 0.038090 | 0.067711 | 0.094121 | 1.196302 | 0.006455 |
| zimg Lanczos3 | 0.027418 [0.022310, 0.031964] | 0.039106 | 0.066527 | 0.084259 | 1.032090 | 0.008585 |
| zimg Spline36 | 0.027342 [0.022255, 0.031872] | 0.038638 | 0.066622 | 0.083192 | 1.077782 | 0.007793 |
| JBU | 0.028275 [0.021439, 0.035689] | 0.039462 | 0.174359 | 0.163284 | 0.804446 | 0.001706 |
| Guided filter | 0.038401 [0.028048, 0.049213] | 0.056964 | 0.204968 | **0.069187** | 1.609309 | 0.003787 |
| Wada EJBF | **0.021545 [0.018061, 0.024611]** | **0.031964** | 0.085918 | 0.097616 | 0.767082 | **0.000067** |
| Korhonen | 0.026185 [0.021473, 0.030628] | 0.035762 | 0.132138 | 0.136112 | 1.038241 | 0.002602 |
| GALOSH-form | 0.030299 [0.025101, 0.035252] | 0.041560 | 0.140985 | 0.413714 | **0.706160** | 0.004451 |
| Plain Jinc3 | 0.025784 [0.021017, 0.030047] | 0.036571 | **0.066400** | 0.077546 | 1.074609 | 0.006111 |
| LGCR algo2 | 0.023283 [0.019231, 0.026881] | 0.032967 | 0.082286 | 0.111584 | 0.976477 | 0.004680 |
| LGCR algo4 | 0.022359 [0.018362, 0.026198] | 0.032154 | 0.096918 | 0.134899 | 0.753299 | 0.003946 |
| LGCR algo6 (primary) | 0.023654 [0.019530, 0.027230] | 0.033737 | 0.079150 | 0.074547 | 0.836910 | 0.006795 |

Wada EJBF is the strongest development-selected external method and has lower
held-out edge MAE than primary algo6. The paired algo6-minus-Wada delta is
+0.002109 [95% CI +0.000381, +0.003706]; algo6 improves only 18.8% of scenes
after averaging their conditions. Algo6 does improve over plain Jinc:
-0.002129 [95% CI -0.003326, -0.000959], with 59.4% of scenes improved. Wada
improves over plain by -0.004239 [95% CI -0.006813, -0.001508].

Algo4 has a lower held-out mean than the preregistered algo6, but switching
would be test-set selection. Its post-hoc delta against Wada is +0.000814 [95%
CI -0.000768, +0.002480], which crosses zero; it remains a secondary result.

### 3.2 Dependence on the true degradation

The aggregate result hides a complete rank reversal (Table 2). Algo6 is better
than Wada under box and triangle downsampling, while Wada is better under
bicubic and the held-out Lanczos family. Every paired CI excludes zero.

**Table 2. Algo6 minus Wada EJBF edge MAE by degradation. Negative favors
algo6.**

| True degradation | Algo6 | Wada EJBF | Paired delta [95% CI] | Scenes improved |
|---|---:|---:|---:|---:|
| Box | 0.021536 | 0.024256 | -0.002720 [-0.003545, -0.001892] | 78.1% |
| Triangle | 0.021119 | 0.024397 | -0.003278 [-0.004558, -0.002040] | 75.0% |
| Bicubic | 0.025155 | 0.019166 | +0.005988 [+0.003475, +0.008302] | 12.5% |
| Lanczos (unseen) | 0.026809 | 0.018361 | +0.008447 [+0.005471, +0.011179] | 12.5% |

This interaction is larger than the aggregate method difference. A single
cross-kernel average is therefore insufficient evidence for a universal
decoder setting.

### 3.3 Guide mismatch and scene conditions

Table 3 separates cases that satisfy or violate the luma-guide assumption.
Wada's Cb/Cr self-guidance allows it to improve isoluminant chroma edges, where
algo6 correctly reduces to plain Jinc but cannot recover missing detail. Both
guided methods damage genuinely soft chroma relative to plain; algo6's
mutual-structure gate limits, but does not remove, that loss.

**Table 3. Edge MAE by synthetic condition.**

| Condition | Scenes | Algo6 | Wada EJBF | Plain Jinc |
|---|---:|---:|---:|---:|
| Co-edge | 12 | 0.027846 | **0.022416** | 0.034009 |
| Isoluminant | 4 | 0.030980 | **0.023912** | 0.030980 |
| Luma only | 4 | 0.000000 | 0.000000 | 0.000000 |
| Misaligned | 4 | 0.030563 | **0.028049** | 0.032180 |
| Ridge | 4 | **0.032441** | 0.032703 | 0.032581 |
| Soft chroma | 4 | 0.011714 | 0.020448 | **0.008501** |

The largest algo6 regrets against plain are all soft-chroma scenes. The worst
is +0.006932 under Lanczos degradation. This failure is included in the
generated report rather than hidden by the aggregate.

### 3.4 Chroma-siting intervention

An eight-scene, box-degraded diagnostic independently crosses true siting and
assumed siting. Table 4 aggregates matched and mismatched cases. Every method
is harmed by incorrect metadata. For algo6, edge MAE rises from 0.024310 to
0.035877 and absolute phase error rises from 0.349506 to 0.692336 pixels.

**Table 4. Matched versus mismatched siting.**

| Method | Matched edge MAE | Mismatched edge MAE | Matched phase px | Mismatched phase px |
|---|---:|---:|---:|---:|
| Plain Jinc | 0.027192 | 0.038198 | 0.373045 | 0.741340 |
| Wada EJBF | 0.027354 | 0.036826 | 0.358807 | **0.669529** |
| LGCR algo6 | **0.024310** | **0.035877** | **0.349506** | 0.692336 |

This intervention concerns a metadata/grid mismatch. It is distinct from a
real displacement between the luma and chroma boundaries inside an otherwise
correctly sited image.

### 3.5 Mechanism ablations

The aggregate Jinc3 ablation does not support a claim that every LGCR component
monotonically improves performance (Table 5). Similarity alone is better than
plain. Adding phase rescue or anisotropy slightly worsens the aggregate mean;
the full algo2 combination recovers a small gain. For algo6, adding affine
credibility alone is neutral to slightly worse, while the full mutual-structure
combination recovers 0.000134.

**Table 5. Held-out edge-MAE ablation.**

| Configuration | Edge MAE |
|---|---:|
| Plain Jinc3 | 0.025784 |
| Similarity only | 0.023308 |
| + phase rescue | 0.023421 |
| + anisotropy | 0.023448 |
| Full algo2 | 0.023283 |
| Algo6 ungated | 0.023778 |
| Algo6 + affine credibility | 0.023788 |
| Full algo6 (+ mutual structure) | 0.023654 |

Because rescue targets an exact-phase failure of separable interpolating
kernels, a separate exploratory diagnostic uses Lanczos3, strict co-edges, and
disables other gates. At horizontally exact source phase, rescue changes MAE
from 0.027195 to 0.025990, a paired delta of -0.001206 [95% CI -0.001746,
-0.000745], with all 12 scenes improved. At half phase, the rescue term is zero
by construction and the maximum output change is exactly 0. This supports the
narrow phase mechanism, not an aggregate Jinc benefit.

### 3.6 Real-domain validation status

The tracked animation and natural-image manifests are empty. Running
`make eval-corpora` therefore emits `Status: INCOMPLETE` and no domain
statistics. The synthetic results establish controlled behavior only. They do
not establish that released animation contains the assumed edge distribution
or that viewers prefer any reconstruction.

## 4. Discussion

### 4.1 Answers to the research questions

First, the profile measures respond to independently injected shift, tangent
alternation, blur, and overshoot. They make failure reports more informative
than PSNR alone. They do not form a mathematically independent decomposition,
so the contribution is operational clarity rather than a new perceptual model.

Second, constrained detail transfer is useful relative to its own signal-only
base, but it is not the strongest overall method in this benchmark. Wada EJBF
is both older and better on the primary held-out endpoint. Reporting that result
changes the value of LGCR: the method becomes one side of a conditional
tradeoff, not evidence that luma-guided constrained transfer supersedes prior
debleeding.

Third, the true degradation family dominates the ranking. Algo6's candidate
kernel stability does not make it invariant to degradation. The strong loss on
bicubic and unseen Lanczos, together with wins on box and triangle, suggests
that estimating the source degradation or selecting a method per scene is more
promising than further tuning one global strength. That selector must be
evaluated on a new holdout because the interaction was discovered here.

Guide mismatch is equally important. Chroma self-guidance lets EJBF reconstruct
isoluminant boundaries, while LGCR's luma-first design cannot. Conversely,
soft-chroma scenes show why forcing a sharp luma edge into chroma is unsafe.
LGCR's gate reduces this failure compared with EJBF but still loses to leaving
the plain reconstruction alone. The co-edge assumption is a boundary condition,
not a universal image prior.

### 4.2 What is novel, and what is not

This work does not claim the first color-bleeding reducer, the first
decoder-side postprocessor, the first luma-guided chroma reconstruction, or the
first cel-inspired artificial evaluation. The literature contradicts all four
claims.

The narrower contribution is the combination of:

- an unknown-degradation, decoder-only stress protocol;
- separate operational boundary measures;
- an explicit siting intervention that does not conflate metadata with content
  misalignment;
- guide-mismatch and regret reporting;
- an open implementation and raw scene-level results;
- a constrained-transfer case study whose primary negative result is retained.

For a GitHub mini-paper, this is meaningful as a reproducible benchmark and
engineering research record. For a venue paper claiming an animation domain,
the missing source-level corpus remains decisive.

### 4.3 Significance for animation restoration

Animation remains a compelling hypothesis because strong flat-color boundaries
make chroma errors visible and make a luma/chroma co-edge prior plausible. The
controlled results also show why that framing is useful: co-edges, isoluminant
color boundaries, soft chroma under line art, ridges, and phase offsets can be
tested separately instead of being mixed into a generic natural-image average.

However, calling animation an "ideal domain" before measuring actual masters
would reverse the required logic. The project now includes a corpus contract
that requires source-level 4:4:4 or lossless frames, license and hash records,
one frame per shot, and work-clustered uncertainty. Until those manifests are
filled, the paper should say "animation-style synthetic content," not
"animation" without qualification.

### 4.4 Limitations

1. **No real animation evidence.** The domain corpus and blinded subjective
   comparison are incomplete. This is the largest limitation.
2. **No codec in the primary forward model.** Quantization, transform coding,
   deblocking, and encoder-specific filters are absent. Wada EJBF was designed
   for both subsampling and compression noise, so this benchmark covers only
   part of its intended task.
3. **Synthetic generator dependence.** Development and test use different
   seeds but the same generator, palette, and condition family. Thirty-two
   scenes do not represent the distribution of released animation.
4. **Simplified sampling kernels.** Box, triangle, bicubic, and Lanczos are
   controlled families, not a catalog of production encoder filters. Boundary
   normalization and siting conventions are implementation choices.
5. **Formula-level baselines.** Wada EJBF, Korhonen, JBU, guided filter, and the
   GALOSH-form method are independent analytic implementations, not author-code
   replications. The GALOSH adaptation also moves a Bayer/CFA method onto an
   H.273-like 4:2:0 grid.
6. **Protocol amendment.** Wada EJBF was added after the original LGCR test run.
   Its own test output was not observed before development selection, but the
   process is not equivalent to a single prospective preregistration.
7. **Multiple diagnostics.** Secondary metrics and strata are unadjusted for
   multiple comparisons. Their confidence intervals are descriptive.
8. **Metrics are signal-domain proxies.** Edge MAE and alpha profiles do not
   model viewing distance, display conversion, or perceptual masking. The
   projection assumes a two-color local boundary.
9. **No temporal or runtime claim.** Animation cadence, repeated frames,
   temporal consistency, implementation speed, and memory use are outside the
   study.
10. **Non-monotonic ablations.** Rescue, anisotropy, and credibility gating do
    not each improve aggregate Jinc3 results. They must not be presented as
    universally beneficial components.

### 4.5 Next evaluation

The next confirmatory study should freeze a licensed 4:4:4 animation corpus, a
matched natural-image corpus, codec settings, and a new test split before any
method selection. It should include Wada EJBF, plain high-quality resampling,
LGCR, and a degradation-aware selector; report both objective profiles and a
blinded paired preference study; and cluster inference by work and shot. A
separate frequency/phase sweep should broaden the current axial rescue and
Nyquist diagnostics. Temporal reconstruction should remain a later extension
until the single-frame claim is stable.

## 5. Conclusion

Chroma boundary repair is not one artifact and not one universally best
filter. In the controlled animation-style benchmark, LGCR algo6 improves its
plain Jinc base but loses the primary comparison to Wada EJBF. The ranking then
reverses across degradation kernels, soft-chroma scenes expose guide risk, and
incorrect siting nearly doubles phase error. These are useful findings because
they replace a broad novelty claim with a falsifiable map of when methods work.

The controlled study is technically ready for a transparent GitHub working-
paper release, but the repository still needs author, license, and citation
metadata. It is not ready to claim effectiveness on released animation. That
claim requires the licensed real-domain and subjective evaluation specified in
the publication plan.

## Reproducibility and Data Availability

Source code and evaluation scripts are hosted at
[MysteryDove/vapoursynth-LGCR](https://github.com/MysteryDove/vapoursynth-LGCR).
The exact protocol is in `evaluation/protocol.md`; parameters are in
`evaluation/config.json`; raw rows and generated reports are in
`evaluation/results/`. Synthetic scenes are generated from fixed seeds and do
not require external data. Real media are intentionally not bundled; their
admission contract and empty manifests are under `evaluation/corpora/`.

## References

1. F.-X. Coudoux, M. Gazalet, and P. Corlay, "An Adaptive Postprocessing
   Technique for the Reduction of Color Bleeding in DCT-Coded Images," IEEE
   TCSVT, 2004. [doi:10.1109/TCSVT.2003.819179](https://doi.org/10.1109/TCSVT.2003.819179)
2. F.-X. Coudoux, M. Gazalet, and P. Corlay, "Reduction of Color Bleeding for
   4:1:1 Compressed Video," IEEE Transactions on Broadcasting, 2005.
   [doi:10.1109/TBC.2005.852243](https://doi.org/10.1109/TBC.2005.852243)
3. A. Catorina et al., "Adaptive Color Bleeding Removal for Video and Still
   DCT Compressed Sequences," SPIE Digital Photography III, 2007.
   [doi:10.1117/12.703061](https://doi.org/10.1117/12.703061)
4. A. Punchihewa and J. Armstrong, "Effects of Sub-Sampling and Quantisation
   on Colour Bleeding Due to Image and Video Compression," IVCNZ, 2008.
   [doi:10.1109/IVCNZ.2008.4762087](https://doi.org/10.1109/IVCNZ.2008.4762087)
5. A. Punchihewa, "Objective Evaluation of Colour Bleeding Artefact Due to
   Image Codecs," VIE, 2008.
   [doi:10.1049/cp:20080420](https://doi.org/10.1049/cp:20080420)
6. S. Li, O. C. Au, L. Sun, W. Dai, and R. Zou, "Color Bleeding Reduction in
   Image and Video Compression," ICCSNT, 2011.
   [doi:10.1109/ICCSNT.2011.6182054](https://doi.org/10.1109/ICCSNT.2011.6182054)
7. N. Wada, M. Kazui, and M. Haseyama, "Extended Joint Bilateral Filter for
   the Reduction of Color Bleeding in Compressed Image and Video," ITE
   Transactions on MTA, 2015. [doi:10.3169/mta.3.95](https://doi.org/10.3169/mta.3.95)
8. J. Kopf et al., "Joint Bilateral Upsampling," ACM TOG, 2007.
   [doi:10.1145/1276377.1276497](https://doi.org/10.1145/1276377.1276497)
9. K. He, J. Sun, and X. Tang, "Guided Image Filtering," IEEE TPAMI, 2013.
   [doi:10.1109/TPAMI.2012.213](https://doi.org/10.1109/TPAMI.2012.213)
10. C. N. Ochotorena and Y. Yamashita, "Anisotropic Guided Filtering," IEEE
    TIP, 2020. [doi:10.1109/TIP.2019.2941326](https://doi.org/10.1109/TIP.2019.2941326)
11. J. Korhonen, "Improving Image Fidelity by Luma-Assisted Chroma
    Subsampling," IEEE ICME, 2015.
    [doi:10.1109/ICME.2015.7177387](https://doi.org/10.1109/ICME.2015.7177387)
12. S. Wang et al., "Joint Chroma Downsampling and Upsampling for Screen
    Content Image," IEEE TCSVT, 2016.
    [doi:10.1109/TCSVT.2015.2461891](https://doi.org/10.1109/TCSVT.2015.2461891)
13. T. Vermeir et al., "Guided Chroma Reconstruction for Screen Content
    Coding," IEEE TCSVT, 2016.
    [doi:10.1109/TCSVT.2015.2469118](https://doi.org/10.1109/TCSVT.2015.2469118)
14. K.-L. Chung, C.-C. Huang, and T.-C. Hsu, "Adaptive Chroma
    Subsampling-Binding and Luma-Guided Chroma Reconstruction Method for Screen
    Content Images," IEEE TIP, 2017.
    [doi:10.1109/TIP.2017.2749148](https://doi.org/10.1109/TIP.2017.2749148)
15. K.-L. Chung, Y.-C. Liang, and C.-S. Wang, "Effective Content-Aware Chroma
    Reconstruction Method for Screen Content Images," IEEE TIP, 2019.
    [doi:10.1109/TIP.2018.2875340](https://doi.org/10.1109/TIP.2018.2875340)
16. Q. Fu et al., "Weighted Chroma Downsampling and Luma-Referenced Chroma
    Upsampling for HDR/WCG Video Coding," IEEE Access, 2019.
    [doi:10.1109/ACCESS.2019.2911673](https://doi.org/10.1109/ACCESS.2019.2911673)
17. Y. Sato, "GALOSH: Blind, Training-Free Denoising of Raw Bayer and sRGB
    Images by Parallel-Friendly Local Shrinkage," arXiv, 2026.
    [doi:10.48550/arXiv.2607.03768](https://doi.org/10.48550/arXiv.2607.03768)
18. ITU-T, "Coding-Independent Code Points for Video Signal Type
    Identification," Recommendation H.273, July 2024.

Full citation metadata is available in [references.bib](references.bib).
