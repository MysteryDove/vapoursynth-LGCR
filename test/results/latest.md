# LGCR battery results

```
case              plain     algo2     algo4     algo6
-----------------------------------------------------
hard_v          0.03630   0.02361   0.02181   0.02450
hard_d45        0.08810   0.07974   0.04133   0.06049
isoluminant     0.03630   0.03630   0.03630   0.03630
misalign4       0.02030   0.02030   0.02030   0.02030
misalign1       0.01558   0.03345   0.03602   0.01731
nullspace_x     0.03630   0.03630   0.03630   0.02450
nullspace_y     0.03630   0.05964   0.05684   0.02463
nullspace_xy    0.03630   0.03630   0.03630   0.02450
hardL_softC     0.00564   0.00592   0.00566   0.00564
ridge_line      0.00181   0.00181   0.00182   0.00187
ramp            0.00505   0.00505   0.00550   0.00505
texture         0.01196   0.01196   0.01196   0.01196
noise           0.03752   0.03758   0.03805   0.02953
upscale2x       0.03860   0.03367   0.03291   0.02505
temporal        0.00008   0.00007   0.00005   0.00004   (aligned edge-band error variance; lower=better)

temporal cases (plain / algo2 / trecon):
t_move1         0.02483   0.01587   0.01598
t_move2         0.02483   0.01587   0.01587
t_move3         0.02483   0.01587   0.01598
t_static_noise   0.02780   0.02846   0.02631
t_single      isolation maxdiff = 0.000000  (must be ~0: single frame has no temporal information)
t_flat_amb      0.01220   0.01220   (identical frames, unobservable motion: trecon ~= algo2 required)

int8 constant (64,128,128): {'recon8': [64, 128, 128], 'trecon8': [64, 128, 128]}  OK
```
