# Evaluation Protocol

Version 1 of this document was frozen before the original held-out results were
generated. Version 2 adds the Wada et al. EJBF baseline after a literature
audit. Its published parameters and finite-support rule were fixed, and the
external comparator is reselected on development scenes, before running EJBF
on held-out scenes. No LGCR method, parameter, or primary-method choice changes
in this amendment. The existing scripts under `test/` remain regression tests
and development history; they are not treated as paper evidence.

## Scope

The study covers training-free, decoder-side, single-frame reconstruction from
4:2:0 to 4:4:4 for animation-style high-contrast boundaries. Temporal
reconstruction, sharpening, cross-color, interlaced chroma errors, and artifacts
invented by downstream neural enhancement are outside the main comparison.

## Forward Model

For full-resolution chroma `C`, degradation kernel `h_D`, true siting `s`, and
codec/quantizer `Q`, the decoder-visible samples are

```text
C420[m,n] = Q((h_D * C)[2m + sx, 2n + sy]) + codec_noise.
```

A method observes `C420` and full-resolution luma `Y`, then reconstructs

```text
Chat = R(C420, Y; assumed_siting, method_parameters).
```

True and assumed siting are independent experimental factors. A metadata/grid
mismatch is not conflated with a real luma/chroma boundary displacement.

## Artifact Definitions

- **Bleeding/blur:** excess transition spread and cross-boundary mixture.
- **Aliasing/staircase:** high-frequency variation of the reconstructed edge
  location along the tangent direction after subtracting the ground-truth edge.
- **Phase shift:** mean normal displacement of the reconstructed 50% crossing.
- **Ringing/halo:** overshoot of the reconstructed chroma after projection onto
  the line between the two ground-truth side colors. Orthogonal color error is
  not counted by this measure.

The primary endpoint is chroma MAE in a four-pixel band around the true chroma
boundary. Secondary endpoints are full-frame MAE/PSNR, smooth-region MAE, phase,
alias, transition-spread delta, bleed-profile error, and projected ringing
mass.

## Splits and Experimental Unit

- Development scenes use seeds 100-111.
- Held-out scenes use seeds 1000-1031.
- Parameter selection uses only development scenes and degradation kernels box,
  triangle, and bicubic.
- Lanczos degradation is held out as an unseen kernel family.
- The independent unit for uncertainty is the scene, not a crop, edge pixel, or
  degradation variant. Bootstrap resampling therefore draws scene identifiers
  and retains every condition belonging to a selected scene.

## Comparisons

Signal-only baselines: zimg bilinear, bicubic, Lanczos3, and Spline36. Analytic
guide-based baselines: canonical Gaussian JBU, guided-filter upsampling, Wada
et al.'s extended joint bilateral filter (EJBF), the Korhonen four-candidate
luma-MSD rule, and an H.273-grid adaptation of GALOSH RAW's signed EWA-Jinc
JBU. EJBF uses the paper's normalized-range parameters and bilinear pre-
upsampling; its spatial Gaussian is truncated at three sigma. These are
formula-level evaluation references, not optimized or bit-exact ports of the
authors' complete systems.

LGCR comparisons: plain Jinc (`strength=0`) and algorithms 2, 4, and 6. The paper
will select one primary LGCR variant on the development set and freeze it before
examining held-out results. No per-scene oracle selection is permitted.

## Statistics and Reporting

For every method, report the scene-weighted mean and a scene bootstrap 95%
confidence interval. Comparisons use paired scene-level deltas. Also report
per-degradation results, the fraction of scenes improved over plain Jinc,
smooth-region regression, and the five largest held-out regrets. Raw per-scene
rows are retained in CSV.

The controlled benchmark establishes mechanism behavior, not performance on the
distribution of released animation. Claims about animation prevalence require a
separate licensed source-level corpus and a natural-image comparison corpus.

## Real-Domain Validation Contract

The separate co-edge prevalence study is governed by the tracked manifests and
`evaluation/corpora/README.md`. Pilot frames may be used to inspect the fixed
edge thresholds but are excluded from reported domain estimates. The test split
requires at least 10 independent works and 30 shots per domain before a domain
comparison is considered complete. Work-level means are the independent units;
animation-minus-natural intervals resample works independently within each
domain. Empty or undersized manifests report `INCOMPLETE` rather than producing
an animation-domain claim.
