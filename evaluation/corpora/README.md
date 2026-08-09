# Domain Corpus Contract

The synthetic benchmark tests reconstruction mechanisms. It does not establish
that co-located luma/chroma edges are more prevalent in animation than in
photographs. The two manifests in this directory reserve that separate domain
validation experiment.

## Admission Rules

- Use source-level, full-chroma RGB stills exported losslessly from the master.
  A decoded 4:2:0 screenshot cannot measure the pre-subsampling edge
  distribution. The current reader accepts Pillow-supported image files,
  converts them to 8-bit RGB, and assumes they have already been normalized to
  sRGB. Export YUV 4:4:4 sources to lossless full-chroma RGB before admission.
- Record an explicit redistribution or research-use license, canonical source
  URL, and SHA-256 digest. Do not commit media unless its license permits it;
  local files belong under `evaluation/corpora/data/`, which is ignored.
- Animation entries must be 2D/cel-style or deliberately flat-shaded. Record
  3D animation separately rather than silently treating it as the target
  domain.
- Natural-image entries must be camera-originated photographs without graphic
  overlays or synthetic compositing.
- Sample at most one frame per shot and record `work_id` and `shot_id`.
  Uncertainty is clustered by work, not by frame or pixel.
- Reserve a pilot subset for threshold inspection. Do not alter the frozen
  thresholds after evaluating the test subset.
- Target at least 10 independent works and 30 shots per domain before making a
  comparative domain claim. A smaller corpus is descriptive only.

## Manifest Fields

`id,domain,split,path,sha256,license,source_url,work_id,shot_id,notes`

Paths are resolved relative to the manifest. `domain` is `animation` or
`natural`; `split` is `pilot` or `test`. Every non-empty row must supply all
fields except `notes`.

Run the validation with:

```sh
make eval-corpora
```

To evaluate one or more alternate manifests without also loading the tracked
defaults, repeat `--manifest` explicitly:

```sh
python3 -m evaluation.coedge --manifest /path/a.csv --manifest /path/b.csv
```

Use `--require-complete` in a publication gate. It fails unless the test split
contains at least 10 works and 30 shots in each domain. Pilot rows are written
to the raw CSV but are always excluded from reported estimates and confidence
intervals.

The runner verifies hashes, converts sRGB to BT.709 YCbCr, detects luma and
chroma boundaries with fixed normalized gradient thresholds, and reports:

- luma- and chroma-edge density;
- fraction of chroma-edge pixels within one pixel of a luma edge;
- fraction of luma-edge pixels within one pixel of a chroma edge;
- absolute gradient-direction agreement on co-located edges.

Empty manifests intentionally produce an `INCOMPLETE` report. This prevents an
unfilled empirical gap from being mistaken for evidence.
