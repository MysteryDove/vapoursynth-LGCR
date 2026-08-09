# Publication Plan

## Target

Publish a compact GitHub technical report titled **From 4:2:0 Sampling to
Color Bleed: A Controlled Study of Chroma Boundary Reconstruction for
Animation-Style Content**. The report is an evaluation and failure-analysis
paper with LGCR as the studied method, not a state-of-the-art claim.

The defensible research gap is:

> Existing work already addresses decoder-side color bleeding, luma-guided
> chroma reconstruction, and cel-like artificial imagery. What remains poorly
> characterized is decoder-only reconstruction under unknown degradation and
> chroma siting, with separate operational measures for boundary blur, tangent
> aliasing, systematic phase displacement, and ringing, plus explicit tests of
> luma/chroma guide mismatch on animation-style structures.

## Frozen Decisions

- Scope: training-free, single-frame, decoder-side 4:2:0 to 4:4:4.
- Primary endpoint: chroma MAE within four pixels of the ground-truth boundary.
- Independent uncertainty unit: synthetic scene.
- Primary LGCR method: `lgcr_algo6`, selected on development scenes.
- Strongest external method: `wada_ejbf`, selected on development scenes after
  the protocol-v2 literature amendment.
- `lgcr_algo4` is not promoted after seeing its lower held-out mean.
- Temporal reconstruction, sharpening, interlacing, cross-color, and neural
  enhancement artifacts are outside the paper.

## Claim Ledger

| Claim | Evidence | Status |
|---|---|---|
| The four artifact labels can be measured separately enough for controlled interventions. | Independent metric-injection tests in `evaluation/test_protocol.py`. | Supported for synthetic profiles; measures are not statistically orthogonal. |
| LGCR algo6 improves on plain Jinc in the controlled benchmark. | Paired scene delta -0.002129, 95% CI [-0.003326, -0.000959]. | Supported. |
| LGCR algo6 beats the strongest existing baseline overall. | Algo6 minus Wada EJBF +0.002109, 95% CI [+0.000381, +0.003706]. | Refuted; must not be claimed. |
| Performance depends on the unknown degradation kernel. | Algo6 beats Wada for box/triangle and loses for bicubic/Lanczos, with scene-bootstrap CIs excluding zero. | Supported on the synthetic kernel set. |
| Correct chroma siting matters independently of guide design. | Controlled matched/mismatched intervention; algo6 phase error 0.350 vs 0.692 px. | Supported on eight synthetic scenes. |
| Phase rescue addresses the separable-kernel exact-phase failure. | Exploratory exact-phase delta -0.001206; half-phase output change exactly zero. | Mechanistically supported; post-hoc and not a confirmatory aggregate claim. |
| LGCR requires Jinc as its reconstruction base. | A development-selected Lanczos4 base beats Jinc3 on 64 new scenes by -0.000945 edge MAE, 95% CI [-0.001161, -0.000733]. | Refuted on the supplemental synthetic holdout; Jinc3 retains lower tangent alias. |
| Wada EJBF is better overall than Lanczos4-based algo6. | On 64 further scenes, Wada-minus-Lanczos4 is -0.000963, 95% CI [-0.002209, +0.000426]. | Inconclusive in aggregate; Wada wins bicubic/Lanczos and loses box/triangle. |
| Wada and LGCR corrections contain complementary information. | The prespecified exploratory hybrid is -0.001777 versus Wada, 95% CI [-0.002606, -0.000965], but improves only 45.3% of scenes and harms soft chroma. | Mechanistically supported on the third synthetic split; not a selected method or real-domain claim. |
| Animation contains more co-located luma/chroma edges than photographs. | Tracked manifests are empty; result says `INCOMPLETE`. | Unsupported and submission-blocking for a real-animation claim. |
| The method improves perceived released animation. | No licensed source corpus or blinded user study. | Unsupported. |

## Remaining Gaps

### Required Before The GitHub Working-Paper Release

1. Choose and add explicit licenses for the source code and paper/evaluation
   artifacts. No license is currently present, so repository visibility alone
   does not grant reuse rights.
2. Create a versioned release and archive that release to a DOI-granting
   repository before replacing the working-paper citation.

### Required Before an Animation-Domain Venue Submission

1. Fill both licensed source-level manifests using at least 10 works and 30
   independent shots per domain, then run the frozen co-edge analysis.
2. Add a real 4:4:4 or lossless-master animation benchmark. If only released
   4:2:0 material is available, report it qualitatively and do not call it
   ground truth.
3. Add codec-inclusive degradation: at minimum JPEG and one video codec at
   multiple chroma QPs. The current primary benchmark isolates subsampling and
   contains no quantization or transform coding.
4. Validate the Wada EJBF reproduction against author code or a hand-checked
   reference implementation. The current baseline follows Eq. 10 and published
   parameters but is not bit-exact author software.
5. Run a preregistered blinded paired comparison on real animation, with crops
   sampled before viewing method outputs.

### Recommended

1. Add runtime and memory comparisons using equally optimized implementations.
2. Expand phase and frequency sweeps beyond axial Nyquist patterns.
3. Estimate the degradation family before choosing between EJBF-like smoothing
   and LGCR-like constrained transfer, and learn a co-edge/soft-chroma gate for
   any hybrid; evaluate both only on a new holdout without oracle labels.
4. Repeat the study in a perceptually better specified color space and report
   chroma-weighted perceptual metrics alongside signal-domain measures.
5. Repeat the base-kernel comparison with radial, anisotropic, and
   codec-estimated point-spread functions; the current separable degradations
   may favor the separable Lanczos4 reconstruction family over radial Jinc.

## Release Checklist

- [x] Freeze development/test split and scene-level bootstrap.
- [x] Add Wada EJBF and disclose the protocol amendment.
- [x] Generate development, held-out, ablation, siting, and phase reports.
- [x] Add a licensed-corpus contract that fails visibly when empty.
- [x] Draft the IMRaD paper and verified bibliography.
- [x] Rewrite the README as an English usage-only guide.
- [x] Add a raw-CSV-to-paper consistency gate.
- [x] Isolate pilot/test corpus rows and add work-level domain contrasts.
- [x] Pass the complete plugin, regression, metric, and paper consistency suite.
- [x] Set the paper and citation author to `MysteryDove`.
- [x] Screen non-Jinc bases and confirm Lanczos4 on a new synthetic holdout.
- [x] Compare Wada EJBF with Lanczos4-algo6 on a third frozen synthetic split
  and retain the exploratory hybrid's soft-chroma failure.
- [x] Freeze working paper version 0.1 as annotated Git tag `paper-v0.1`.
- [ ] Add explicit repository licenses.
- [ ] Populate and run the real-domain corpus.
- [ ] Add codec-inclusive and subjective studies.
- [ ] Replace the paper's working-paper warning only after the blocking gaps are
  complete.
