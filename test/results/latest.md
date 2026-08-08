# LGCR battery results

```
case              plain     algo2     algo1     algo3     algo4     algo5
-------------------------------------------------------------------------
hard_v          0.03630   0.02361   0.02361   0.01687   0.02181   0.05767
hard_d45        0.08810   0.07974   0.08861   0.04133   0.04133   0.09679
isoluminant     0.03630   0.03630   0.03630   0.03630   0.03630   0.05767
misalign4       0.02030   0.02030   0.01882   0.02030   0.02030   0.04328
hardL_softC     0.00564   0.00592   0.01522   0.00563   0.00566   0.02505
ridge_line      0.00181   0.00181   0.00330   0.00515   0.00182   0.00569
ramp            0.00505   0.00505   0.00505   0.00823   0.00550   0.02228
texture         0.01196   0.01196   0.01196   0.01196   0.01196   0.03135
noise           0.03752   0.03758   0.03656   0.04888   0.03805   0.05856
upscale2x       0.03860   0.03367   0.03167   0.02271   0.03291       n/a
temporal        0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   (inter-frame diff in static region; lower=better)

temporal cases (plain / algo2 / trecon):
t_move1         0.02483   0.01587   0.01598
t_move2         0.02483   0.01587   0.01587
t_move3         0.02483   0.01587   0.01598
t_static_noise   0.02780   0.02846   0.02631
t_single      isolation maxdiff = 0.000000  (must be ~0: single frame has no temporal information)
t_flat_amb      0.01220   0.01220   (identical frames, unobservable motion: trecon ~= algo2 required)

int8 constant (64,128,128): {'recon8': [64, 128, 128], 'trecon8': [64, 128, 128]}  OK
```
