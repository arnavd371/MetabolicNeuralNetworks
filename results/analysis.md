## Part 1: per-seed reproduction vs repo

| batch   |   seed |   easy_to_med |   med_to_hard |   acc |   repo_easy |   repo_hard |   diff_easy |   diff_hard |
|:--------|-------:|--------------:|--------------:|------:|------------:|------------:|------------:|------------:|
| batch1  |     42 |             7 |            18 | 74.08 |           7 |          18 |           0 |           0 |
| batch1  |      7 |             2 |            28 | 75.28 |           2 |          28 |           0 |           0 |
| batch1  |    123 |             4 |            34 | 75.34 |           4 |          34 |           0 |           0 |
| batch1  |      0 |             0 |            31 | 75.09 |           0 |          31 |           0 |           0 |
| batch1  |     99 |             8 |            24 | 75.98 |           8 |          24 |           0 |           0 |
| batch1  |     11 |             9 |            18 | 74.75 |           9 |          18 |           0 |           0 |
| batch1  |     22 |            15 |            20 | 77.16 |          15 |          20 |           0 |           0 |
| batch1  |     33 |             6 |            34 | 74.25 |           6 |          34 |           0 |           0 |
| batch1  |     44 |             8 |            39 | 76.46 |           8 |          39 |           0 |           0 |
| batch1  |     55 |             1 |            22 | 75.99 |           1 |          22 |           0 |           0 |
| batch2  |    100 |            12 |            24 | 77.98 |          12 |          24 |           0 |           0 |
| batch2  |    101 |             0 |            29 | 76.72 |           0 |          29 |           0 |           0 |
| batch2  |    102 |             4 |            25 | 74.84 |           4 |          25 |           0 |           0 |
| batch2  |    103 |             2 |            23 | 76.76 |           2 |          23 |           0 |           0 |
| batch2  |    104 |             0 |            46 | 74.42 |           0 |          46 |           0 |           0 |
| batch2  |    105 |             0 |            33 | 77.69 |           0 |          33 |           0 |           0 |
| batch2  |    106 |            13 |            25 | 77.1  |          13 |          25 |           0 |           0 |
| batch2  |    107 |             0 |            35 | 74.06 |           0 |          35 |           0 |           0 |
| batch2  |    108 |             7 |            32 | 74.85 |           7 |          32 |           0 |           0 |
| batch2  |    109 |             4 |            16 | 74.43 |           4 |          16 |           0 |           0 |

### Part 1 statistics (my reproduction)

| comparison | A | B | A mean ± SD | B mean ± SD | B/A ratio [95% CI] | t | p | Wilcoxon p | d | B>A / ties / B<A |
|---|---|---|---|---|---|---|---|---|---|---|
| Part1 batch1 | E→M | M→H | 6.00 ± 4.24 | 26.80 ± 7.10 | 4.47× [2.78, 8.11] | 6.653 | 9.34e-05 | 0.0020 | 2.10 | 10/0/0 (n=10) |
| Part1 batch2 | E→M | M→H | 4.20 ± 4.71 | 28.80 ± 7.82 | 6.86× [3.57, 20.80] | 6.857 | 7.42e-05 | 0.0020 | 2.17 | 10/0/0 (n=10) |
| Part1 pooled | E→M | M→H | 5.10 ± 4.57 | 27.80 ± 7.53 | 5.45× [3.64, 9.25] | 9.633 | 9.58e-09 | 8.75e-05 | 2.15 | 20/0/0 (n=20) |

## Part 2: check schedule (logged, original curriculum)

| phase   |   epoch |   start_step |   end_step |   n_steps |   n_checks |
|:--------|--------:|-------------:|-----------:|----------:|-----------:|
| Easy    |       0 |            1 |        157 |       157 |          1 |
| Easy    |       1 |          158 |        314 |       157 |          1 |
| Medium  |       0 |          786 |       1020 |       235 |          1 |
| Medium  |       1 |         1021 |       1255 |       235 |          2 |
| Hard    |       0 |         1961 |       2351 |       391 |          2 |
| Hard    |       1 |         2352 |       2742 |       391 |          3 |

Check steps (first 4 per phase/epoch):

| phase   |   epoch |   checks_per_seed | check_steps        |
|:--------|--------:|------------------:|:-------------------|
| Easy    |       0 |                 1 | [150]              |
| Easy    |       1 |                 1 | [300]              |
| Hard    |       0 |                 2 | [2100, 2250]       |
| Hard    |       1 |                 3 | [2400, 2550, 2700] |
| Medium  |       0 |                 1 | [900]              |
| Medium  |       1 |                 2 | [1050, 1200]       |

### Part 2a/2b: original curriculum: raw, per-check and fixed-window

| comparison | A | B | A mean ± SD | B mean ± SD | B/A ratio [95% CI] | t | p | Wilcoxon p | d | B>A / ties / B<A |
|---|---|---|---|---|---|---|---|---|---|---|
| orig: epoch-0 deaths | Medium | Hard | 5.10 ± 4.57 | 27.80 ± 7.53 | 5.45× [3.63, 9.30] | 9.633 | 9.58e-09 | 8.75e-05 | 2.15 | 20/0/0 (n=20) |
| orig: deaths per check, epoch 0 | Medium | Hard | 5.10 ± 4.57 | 13.90 ± 3.77 | 2.73× [1.82, 4.65] | 5.479 | 2.76e-05 | 5.35e-04 | 1.23 | 16/2/2 (n=20) |
| orig: first 150 steps | Medium | Hard | 5.10 ± 4.57 | 7.20 ± 4.71 | 1.41× [0.84, 2.66] | 1.209 | 0.2414 | 0.3082 | 0.27 | 10/3/7 (n=20) |
| orig: first 300 steps | Medium | Hard | 41.95 ± 10.84 | 27.80 ± 7.53 | 0.66× [0.55, 0.80] | -4.061 | 6.67e-04 | 0.0011 | -0.91 | 2/0/18 (n=20) |
| orig: Medium check#1 vs Hard check#1 (epoch 0) | Medium c1 | Hard c1 | 5.10 ± 4.57 | 7.20 ± 4.71 | 1.41× [0.84, 2.66] | 1.209 | 0.2414 | 0.3082 | 0.27 | 10/3/7 (n=20) |
| orig: Medium check#1 vs Hard check#2 (epoch 0) | Medium c1 | Hard c2 | 5.10 ± 4.57 | 20.60 ± 4.73 | 4.04× [2.74, 6.77] | 8.961 | 2.99e-08 | 8.70e-05 | 2.00 | 20/0/0 (n=20) |

### Part 2c: no-shift baselines in the same runs

| comparison | A | B | A mean ± SD | B mean ± SD | B/A ratio [95% CI] | t | p | Wilcoxon p | d | B>A / ties / B<A |
|---|---|---|---|---|---|---|---|---|---|---|
| Medium: per-check, epoch 1 (A) vs epoch 0 (B) | Medium ep1 | Medium ep0 | 20.43 ± 4.76 | 5.10 ± 4.57 | 0.25× [0.16, 0.34] | -14.288 | 1.29e-11 | 8.76e-05 | -3.19 | 0/0/20 (n=20) |
| Medium: mid-phase 300 (A) vs first 300 (B) | Medium mid300 | Medium first300 | 5.70 ± 5.13 | 41.95 ± 10.84 | 7.36× [5.17, 12.00] | 13.384 | 4.02e-11 | 8.81e-05 | 2.99 | 20/0/0 (n=20) |
| Hard: per-check, epoch 1 (A) vs epoch 0 (B) | Hard ep1 | Hard ep0 | 4.05 ± 2.11 | 13.90 ± 3.77 | 3.43× [2.69, 4.47] | 10.259 | 3.48e-09 | 1.91e-06 | 2.29 | 20/0/0 (n=20) |
| Hard: mid-phase 300 (A) vs first 300 (B) | Hard mid300 | Hard first300 | 10.00 ± 6.28 | 27.80 ± 7.53 | 2.78× [2.14, 3.82] | 8.710 | 4.64e-08 | 1.30e-04 | 1.95 | 19/1/0 (n=20) |
| excess deaths (first300 − mid300): Medium (A) vs Hard (B) | Med excess | Hard excess | 36.25 ± 11.81 | 17.80 ± 8.91 | 0.49× [0.35, 0.66] | -4.438 | 2.83e-04 | 7.26e-04 | -0.99 | 2/0/18 (n=20) |

### Part 2d: control curricula (A = first transition, B = second transition)

| comparison | A | B | A mean ± SD | B mean ± SD | B/A ratio [95% CI] | t | p | Wilcoxon p | d | B>A / ties / B<A |
|---|---|---|---|---|---|---|---|---|---|---|
| noshift_matched (4→4→4, original timing): epoch-0 deaths | Medium | Hard | 5.15 ± 4.60 | 12.80 ± 5.69 | 2.49× [1.67, 4.21] | 4.491 | 2.50e-04 | 0.0010 | 1.00 | 16/0/4 (n=20) |
| noshift_matched (4→4→4, original timing): deaths per check, epoch 0 | Medium | Hard | 5.15 ± 4.60 | 6.40 ± 2.84 | 1.24× [0.84, 2.10] | 0.994 | 0.3326 | 0.3314 | 0.22 | 11/0/9 (n=20) |
| noshift_matched (4→4→4, original timing): first 150 steps | Medium | Hard | 5.15 ± 4.60 | 6.40 ± 3.85 | 1.24× [0.75, 2.24] | 0.808 | 0.4292 | 0.3330 | 0.18 | 11/1/8 (n=20) |
| noshift_matched (4→4→4, original timing): first 300 steps | Medium | Hard | 8.45 ± 6.15 | 12.80 ± 5.69 | 1.51× [1.06, 2.33] | 2.158 | 0.0440 | 0.0475 | 0.48 | 13/0/7 (n=20) |
| Hard-slot epoch-0 deaths: no shift (A) vs real 6→10 shift (B) | no shift | 6→10 | 12.80 ± 5.69 | 27.80 ± 7.53 | 2.17× [1.81, 2.67] | 8.385 | 8.28e-08 | 1.19e-04 | 1.87 | 19/0/1 (n=20) |
| Medium-slot epoch-0 deaths: no shift (A) vs real 4→6 shift (B) | no shift | 4→6 | 5.15 ± 4.60 | 5.10 ± 4.57 | 0.99× [0.96, 1.02] | -0.567 | 0.5770 | 0.5637 | -0.13 | 1/17/2 (n=20) |
| excess over matched no-shift run, first 150 steps: 4→6 (A) vs 6→10 (B) | 4→6 excess | 6→10 excess | -0.05 ± 0.38 | 0.80 ± 4.86 | -16.00× [-450000.00, 2150000.00] | 0.756 | 0.4590 | 0.5695 | 0.17 | 11/1/8 (n=20) |
| excess over matched no-shift run, first 300 steps: 4→6 (A) vs 6→10 (B) | 4→6 excess | 6→10 excess | 33.50 ± 7.63 | 15.00 ± 7.80 | 0.45× [0.34, 0.56] | -7.489 | 4.41e-07 | 1.20e-04 | -1.67 | 1/0/19 (n=20) |
| orig: deaths / alive at phase start, first 150 steps | Medium | Hard | 0.07 ± 0.06 | 0.16 ± 0.09 | 2.41× [1.51, 4.30] | 3.383 | 0.0031 | 0.0042 | 0.76 | 14/0/6 (n=20) |
| orig: deaths / alive at phase start, first 300 steps | Medium | Hard | 0.58 ± 0.09 | 0.65 ± 0.10 | 1.12× [1.01, 1.23] | 2.148 | 0.0448 | 0.0484 | 0.48 | 13/0/7 (n=20) |
| orig: eligible fraction at first full post-shift check (step 1050 vs 2250) | Medium | Hard | 0.51 ± 0.06 | 0.63 ± 0.06 | 1.22× [1.14, 1.29] | 6.082 | 7.56e-06 | 1.91e-05 | 1.36 | 18/0/2 (n=20) |
| same, no-shift matched run (timing control) | Medium slot | Hard slot | 0.05 ± 0.05 | 0.08 ± 0.05 | 1.83× [1.14, 3.30] | 2.465 | 0.0234 | 0.0400 | 0.55 | 14/0/6 (n=20) |
| same, excess over no-shift run | 4→6 | 6→10 | 0.47 ± 0.06 | 0.54 ± 0.07 | 1.16× [1.08, 1.23] | 4.170 | 5.20e-04 | 5.86e-04 | 0.93 | 16/0/4 (n=20) |
| equal_len (4→6→10, 157 steps/epoch): epoch-0 deaths | Medium | Hard | 5.40 ± 3.35 | 4.30 ± 4.09 | 0.80× [0.49, 1.16] | -1.109 | 0.2812 | 0.3041 | -0.25 | 7/2/11 (n=20) |
| equal_len (4→6→10, 157 steps/epoch): deaths per check, epoch 0 | Medium | Hard | 5.40 ± 3.35 | 4.30 ± 4.09 | 0.80× [0.49, 1.16] | -1.109 | 0.2812 | 0.3041 | -0.25 | 7/2/11 (n=20) |
| equal_len (4→6→10, 157 steps/epoch): first 150 steps | Medium | Hard | 5.40 ± 3.35 | 4.30 ± 4.09 | 0.80× [0.49, 1.16] | -1.109 | 0.2812 | 0.3041 | -0.25 | 7/2/11 (n=20) |
| equal_len (4→6→10, 157 steps/epoch): first 300 steps | Medium | Hard | 40.95 ± 8.15 | 25.50 ± 7.49 | 0.62× [0.53, 0.73] | -5.865 | 1.20e-05 | 2.91e-04 | -1.31 | 2/1/17 (n=20) |
| reversed (10→6→4): epoch-0 deaths | Medium | Easy | 6.95 ± 3.12 | 6.00 ± 4.01 | 0.86× [0.66, 1.09] | -1.155 | 0.2623 | 0.2643 | -0.26 | 5/3/12 (n=20) |
| reversed (10→6→4): deaths per check, epoch 0 | Medium | Easy | 6.95 ± 3.12 | 6.00 ± 4.01 | 0.86× [0.66, 1.09] | -1.155 | 0.2623 | 0.2643 | -0.26 | 5/3/12 (n=20) |
| reversed (10→6→4): first 150 steps | Medium | Easy | 6.95 ± 3.12 | 6.00 ± 4.01 | 0.86× [0.66, 1.09] | -1.155 | 0.2623 | 0.2643 | -0.26 | 5/3/12 (n=20) |
| reversed (10→6→4): first 300 steps | Medium | Easy | 11.15 ± 4.08 | 11.40 ± 5.30 | 1.02× [0.82, 1.26] | 0.197 | 0.8461 | 0.8784 | 0.04 | 9/2/9 (n=20) |

Checks in epoch 0 per curriculum: {"orig: checks in epoch 0": {"Medium": [1], "Hard": [2]}, "noshift_matched (4\u21924\u21924, original timing): checks in epoch 0": {"Medium": [1], "Hard": [2]}, "equal_len (4\u21926\u219210, 157 steps/epoch): checks in epoch 0": {"Medium": [1], "Hard": [1]}, "reversed (10\u21926\u21924): checks in epoch 0": {"Medium": [1], "Easy": [1]}}

### Final test accuracy by curriculum

| curriculum      |   mean |   std |   count |
|:----------------|-------:|------:|--------:|
| equal_len       |  72.21 |  1.35 |      20 |
| noshift_matched |  35.19 |  0.37 |      20 |
| original        |  75.66 |  1.25 |      20 |
| reversed        |  44.3  |  2.7  |      20 |

### Part 3: binary per-pass activation indicator (original curriculum)

| comparison | A | B | A mean ± SD | B mean ± SD | B/A ratio [95% CI] | t | p | Wilcoxon p | d | B>A / ties / B<A |
|---|---|---|---|---|---|---|---|---|---|---|
| binary rate: epoch-0 deaths | Medium | Hard | 4.90 ± 3.56 | 28.00 ± 5.98 | 5.71× [3.74, 10.78] | 9.156 | 7.42e-06 | 0.0020 | 2.90 | 10/0/0 (n=10) |
| binary rate: deaths per check, epoch 0 | Medium | Hard | 4.90 ± 3.56 | 14.00 ± 2.99 | 2.86× [1.87, 5.39] | 5.350 | 4.62e-04 | 0.0020 | 1.69 | 10/0/0 (n=10) |
| binary rate: first 150 steps | Medium | Hard | 4.90 ± 3.56 | 4.10 ± 2.91 | 0.84× [0.43, 1.76] | -0.503 | 0.6270 | 0.5469 | -0.16 | 4/1/5 (n=10) |
| binary rate: first 300 steps | Medium | Hard | 41.20 ± 7.90 | 28.00 ± 5.98 | 0.68× [0.54, 0.86] | -3.143 | 0.0119 | 0.0137 | -0.99 | 2/0/8 (n=10) |
| frac rate (Part 2 runs, same 10 seeds): epoch-0 deaths | Medium | Hard | 6.00 ± 4.24 | 26.80 ± 7.10 | 4.47× [2.78, 8.11] | 6.653 | 9.34e-05 | 0.0020 | 2.10 | 10/0/0 (n=10) |

Binary-rate runs: total deaths per seed over the whole run: {0: 303, 7: 247, 11: 259, 22: 298, 33: 233, 42: 295, 44: 305, 55: 290, 99: 298, 123: 266}; mean final accuracy 76.04%; alive at end [43, 31, 42, 45, 40, 29, 60, 45, 53, 58]

Mean activation rate of alive neurons at checks (binary): 0.734 (min 0.306); alive neurons with r<0.05 before a check, max over all checks: 71
