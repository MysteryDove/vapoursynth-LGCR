# Non-Jinc Base-Kernel Study Protocol

## Status

This is an exploratory development screen followed, only if warranted, by a
new confirmatory holdout. It was defined after the main paper's held-out results
were known. It cannot change the frozen `lgcr_algo6` plus Jinc3 primary result.

## Development Screen

- Scenes: the existing 12 development scenes, seeds 100-111.
- Degradations: box, triangle, and bicubic; left siting only.
- Endpoint: scene-weighted edge MAE.
- Modes: signal-only base and LGCR algo6 using the same base kernel.
- Fixed LGCR settings: strength 0.8, dense evaluation, zero hull margin, full
  affine credibility, and full mutual-structure gating.
- Candidates: Bilinear; Catmull-Rom, sharp, and Mitchell-Netravali bicubic;
  Spline16/36; Lanczos with 2/3/4 lobes; and Jinc with 2/3/4 lobes.
- Selection rule: the non-Jinc candidate with the lowest mean algo6 edge MAE is
  frozen for any new-holdout comparison. Artifact metrics are diagnostic and do
  not alter selection.

## Frozen Supplemental Confirmation

The development rule selected Lanczos4 (`kernel="lanczos", taps=4`) with an
algo6 edge MAE of 0.023005. Jinc3 scored 0.023963. Before evaluating any new
scene, the following confirmation was frozen:

- new scenes: seeds 3000-3063 (64 scenes), not used elsewhere;
- degradations: box, triangle, bicubic, and Lanczos;
- true and assumed siting: matched left and center;
- methods: signal-only and algo6 with either Lanczos4 or Jinc3;
- fixed LGCR settings: strength 0.8, dense evaluation, zero hull margin, full
  affine credibility, and full mutual-structure gating;
- primary endpoint: scene-weighted edge MAE;
- primary comparison: Lanczos4 algo6 minus Jinc3 algo6;
- secondary comparison: Lanczos4 algo6 minus signal-only Lanczos4;
- inference: paired scene bootstrap with 4,000 samples and seed 20260808;
- degradation, condition, and artifact measures: descriptive diagnostics.

Seeds 1000-1031 remain excluded because that main-paper test split has already
been observed. This repository-local freeze is prospective relative to the new
rows, but it is not an externally registered protocol and cannot replace the
main paper's original primary analysis.

This comparison answers whether a different interpolation base improves the
same constrained-transfer method. It does not answer whether Wada EJBF, a new
inverse model, or a learned reconstructor is preferable.
