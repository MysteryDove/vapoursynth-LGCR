# Wada EJBF Supplemental Study Protocol

## Status

This comparison was specified after the main benchmark and the non-Jinc kernel
study were observed. It uses a third, previously unevaluated synthetic scene
set. It is prospective only with respect to these new rows, is not externally
preregistered, and cannot change either earlier primary analysis.

## Frozen Comparison

Before evaluating seeds 4000-4063, the following choices were fixed:

- scenes: 64 new animation-style synthetic scenes;
- degradations: box, triangle, bicubic, and Lanczos;
- true and assumed siting: matched left and center;
- primary endpoint: scene-weighted edge MAE;
- primary comparison: Wada EJBF minus LGCR algo6 with a Lanczos4 base;
- secondary methods: Wada's bilinear input surface, plain Lanczos4, and LGCR
  algo6 with a Jinc3 base;
- Wada parameters: the published normalized-range values already frozen in the
  main paper, namely spatial sigma 4, luma sigma 0.03, joint Cb/Cr sigma 0.10,
  and a three-sigma radius of 12;
- LGCR parameters: strength 0.8, dense evaluation, zero hull margin, full
  affine-credibility gating, and full mutual-structure gating;
- inference: paired scene bootstrap with 4,000 samples and seed 20260808;
- artifact metrics, degradation strata, scene-condition strata, and all hybrid
  results: descriptive secondary analyses.

## Exploratory Hybrid

Wada EJBF is a complete reconstruction method, not an interpolation kernel, so
it cannot be substituted directly for `kernel=`. A deliberately simple hybrid
tests whether the methods contain complementary information:

```text
Wada + [LGCR-algo6(Lanczos4, unclamped) - Plain-Lanczos4(unclamped)]
```

The result is clamped to a bilinearly expanded local 5 by 5 chroma hull. This
hybrid was specified before its rows were evaluated, but it is exploratory: it
does not prove that the extracted correction is the optimal way to combine the
estimators, and it will not be promoted over the primary comparison based on
this split.

The implementation is a formula-level reproduction of Wada et al.'s Eq. 10,
not a bit-exact run of author software. The forward degradations remain
synthetic, separable, and free of transform coding or quantization.
