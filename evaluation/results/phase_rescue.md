# Targeted Phase-Rescue Diagnostic (Exploratory)

Strict co-edge scenes: 12; raw rows: 96.

This post-hoc mechanism diagnostic is not a confirmatory benchmark. It
uses left-sited 4:2:0, separable Lanczos3 reconstruction, and disables
anisotropy, ridge, and mutual-structure gates. At the horizontal exact
phase, a separable interpolating kernel is a delta; at half phase the
rescue term is zero by construction. Deltas are rescue-on minus rescue-off,
so negative values favor rescue. Confidence intervals resample scenes.

| horizontal source phase | rescue off MAE | rescue on MAE | paired delta [95% CI] | scenes improved | max output change |
|---|---:|---:|---:|---:|---:|
| exact | 0.027195 | 0.025990 | -0.001206 [-0.001746, -0.000745] | 100.0% | 0.22478357 |
| half | 0.025442 | 0.025442 | +0.000000 [+0.000000, +0.000000] | 0.0% | 0.00000000 |

## Exact-Phase Delta by Degradation

| degradation | rescue-on minus rescue-off MAE |
|---|---:|
| box | -0.003743 |
| triangle | -0.000554 |
| bicubic | -0.000273 |
| lanczos | -0.000252 |
