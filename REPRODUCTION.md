# MetNN Experiment 1: Reproduction and Confound Analysis

- Repository: github.com/arnavd371/MetabolicNeuralNetworks (commit 72809f2), Experiment1.ipynb
- Hardware: Kaggle, NVIDIA Tesla T4 (driver 580.178), PyTorch 2.11.0+cu128, cuDNN 9.19, Python 3.13.15
- Runs: 110 full training runs (Part 1: 20, Part 2: 80, Part 3: 10). All runs reported; none excluded.


## Summary

1. Reproduction. The original code reproduces exactly. All 20 seeds give death counts identical to the
   repository's saved output, and every aggregate statistic matches (pooled t = 9.633, d = 2.154,
   ratio 5.45x, bootstrap 95% CI [3.64, 9.25], Stouffer z = 5.565, 20/20 seeds consistent).

2. Validity. The 5.45x ratio is mostly a consequence of the counting window, not of shift severity.
   A neuron can only be removed once its 150-step activation-rate window is (almost) entirely
   post-shift, so each transition's death burst appears at the second reorganize() check after it
   (steps 1050 and 2250). The Easy->Medium burst (mean 36.9 deaths) falls just after Medium epoch 0
   ends and is not counted. The Medium->Hard burst (mean 20.6 deaths) falls inside Hard epoch 0.

3. Under controls:
   - With no distribution shift but identical timing, the paper's metric still gives 2.49x (p = 2.5e-4).
   - Medium epoch-0 deaths are identical with and without the 4->6 class change in 17/20 seeds.
   - With equal-length counting windows the effect vanishes (first 150 steps: 1.41x, p = 0.24)
   or reverses (first 300 steps: 0.66x, p = 7e-4; Medium > Hard in 18/20 seeds).
   - With equal epoch lengths across phases the effect vanishes (0.80x, p = 0.28).

4. What survives. Transitions that add classes do produce death bursts relative to a no-shift
   baseline. In absolute counts the 4->6 transition produces more deaths than 6->10. Normalised by
   the alive population (71 vs 42 neurons at the two transitions), the 6->10 shift makes 54% of
   alive neurons eligible for death above the no-shift baseline versus 47% for 4->6: ratio 1.16x,
   95% CI [1.08, 1.23], p = 5e-4, 16/20 seeds. This is a modest effect, far from ~5x.

5. Rate definition. A binary per-pass activation indicator (as described in the paper) gives the
   same qualitative results as the code's fractional rate: 5.71x on the paper's metric, and the
   same disappearance/reversal under fixed windows.


## Statistical Conventions

All comparisons are paired across seeds. "Mean +/- SD" uses the population SD (numpy std, ddof = 0),
matching the original code. Tests: paired t-test, Wilcoxon signed-rank test, Cohen's d computed as
mean(difference) / SD(difference, ddof = 1), bootstrap 95% CI on the ratio of means (10,000 paired
resamples, seed 0), and consistency = number of seeds where B > A / ties / B < A.
In each comparison, A is the first-listed condition and B the second; ratio = mean(B) / mean(A).


## Part 1: Faithful Reproduction

The notebook code cell was extracted verbatim into metnn_exp1.py and run with the README protocol:
run("smoke"); run("exp1", seeds=[42,7,123,0,99,11,22,33,44,55], batch_name="batch1");
run("exp1", seeds=[100..109], batch_name="batch2"); run("meta"); run("figures"); run("summary").
Deterministic cuDNN settings, AMP and set_seed were unchanged.

Per-seed results (Easy->Medium deaths, Medium->Hard deaths, test accuracy). Repo values in brackets.

```text
Batch 1
seed  42:   7 [7]    18 [18]   74.08%
seed   7:   2 [2]    28 [28]   75.28%
seed 123:   4 [4]    34 [34]   75.34%
seed   0:   0 [0]    31 [31]   75.09%
seed  99:   8 [8]    24 [24]   75.98%
seed  11:   9 [9]    18 [18]   74.75%
seed  22:  15 [15]   20 [20]   77.16%
seed  33:   6 [6]    34 [34]   74.25%
seed  44:   8 [8]    39 [39]   76.46%
seed  55:   1 [1]    22 [22]   75.99%
```

```text
Batch 2
seed 100:  12 [12]   24 [24]   77.98%
seed 101:   0 [0]    29 [29]   76.72%
seed 102:   4 [4]    25 [25]   74.84%
seed 103:   2 [2]    23 [23]   76.76%
seed 104:   0 [0]    46 [46]   74.42%
seed 105:   0 [0]    33 [33]   77.69%
seed 106:  13 [13]   25 [25]   77.10%
seed 107:   0 [0]    35 [35]   74.06%
seed 108:   7 [7]    32 [32]   74.85%
seed 109:   4 [4]    16 [16]   74.43%
```

Exact seed-level match: 20/20.

Aggregates (paper / repo output / this reproduction; all three are identical unless noted):

- Batch 1: Easy->Medium 6.0 +/- 4.2, Medium->Hard 26.8 +/- 7.1, ratio 4.47x, CI [2.78, 8.11],
  t = 6.653, p = 9.3e-5, Wilcoxon p = 0.0020, d = 2.104, 10/10 consistent, accuracy 75.44%.
- Batch 2: Easy->Medium 4.2 +/- 4.7, Medium->Hard 28.8 +/- 7.8, ratio 6.86x, CI [3.57, 20.80],
  t = 6.857, p = 7.4e-5, Wilcoxon p = 0.0020, d = 2.168, 10/10 consistent, accuracy 75.88%.
- Pooled (n = 20): Easy->Medium 5.1 +/- 4.57, Medium->Hard 27.8 +/- 7.53, ratio 5.45x,
  CI [3.64, 9.25], t = 9.633, p = 9.6e-9, Wilcoxon p = 8.8e-5, d = 2.154,
  Stouffer combined z = 5.565, 20/20 consistent.

Notes:

- The README reports pooled SDs of 4.4 and 7.4. These are averages of the two batch SDs; the
  population SD over all 20 seeds is 4.57 and 7.53.
- Agreement was exact on a Tesla T4 with the software versions listed above. The original GPU and
  software versions are not recorded; other hardware may not reproduce seed-level values exactly.
- The bootstrap CI depends on the order of the seeds in the input because its random seed is
  fixed; re-ordering the same 20 pairs gives [3.63, 9.30].


## Part 2: Confound Checks

Method. An instrumented copy of _run_curriculum was created. Its training loop is identical line for
line to the original and adds only read-only logging (no extra random-number draws). At every
reorganize() check it logs the global step, phase, epoch, deaths, births, alive count before and
after, the number of alive neurons with rate below theta_death (eligible for death) before the check,
and how many of the 150 steps in the rate window belong to the current phase. Its per-seed
transition deaths match Part 1 in 20/20 seeds, so the logs describe the same runs.

Verified check schedule (identical in every seed; batch size 128, C = 150):
```text
Easy   epoch 0: steps    1-157    checks at 150
Medium epoch 0: steps  786-1020   check at 900 only (rate window: 115 Medium + 35 Easy steps)
Medium epoch 1: steps 1021-1255   checks at 1050, 1200
Hard   epoch 0: steps 1961-2351   checks at 2100, 2250
Hard   epoch 1: steps 2352-2742   checks at 2400, 2550, 2700
```

Mean deaths at the checks around each transition (20 seeds):
```text
step  900 (Medium, 115 post-shift steps in window):  5.10 deaths, 71.1 alive
step 1050 (Medium, first full post-shift window):   36.85 deaths, 71.0 alive
step 2100 (Hard, 140 post-shift steps in window):     7.20 deaths, 42.0 alive
step 2250 (Hard, first full post-shift window):      20.60 deaths, 36.8 alive (23.1 eligible)
step 2400 (Hard):                                      3.15 deaths, 19.6 alive
```

The N_min = 16 floor prevented eligible neurons from being removed in 100 of 780 checks, including
16/20 seeds at step 2250, so Hard-phase death counts are capped by the population floor.

### (a) Per-check analysis

- Epoch-0 deaths (paper metric): Medium 5.10 +/- 4.57, Hard 27.80 +/- 7.53; ratio 5.45x [3.63, 9.30];
  t = 9.63, p = 9.6e-9; Wilcoxon p = 8.8e-5; d = 2.15; 20/0/0.
- Epoch-0 deaths per check: Medium 5.10 +/- 4.57, Hard 13.90 +/- 3.77; ratio 2.73x [1.82, 4.65];
  t = 5.48, p = 2.8e-5; Wilcoxon p = 5.4e-4; d = 1.23; 16/2/2.
- First check after each transition (step 900 vs 2100): Medium 5.10 +/- 4.57, Hard 7.20 +/- 4.71;
  ratio 1.41x [0.84, 2.66]; t = 1.21, p = 0.24; Wilcoxon p = 0.31; d = 0.27; 10/3/7.
- Medium check 1 vs Hard check 2 (step 900 vs 2250): 5.10 vs 20.60; ratio 4.04x [2.74, 6.77];
  p = 3.0e-8; d = 2.00; 20/0/0.

### (b) Fixed step windows aligned to the transition

Any 150-step window contains exactly one check and any 300-step window exactly two, so both
transitions get equal opportunity.

- First 150 steps: Medium 5.10 +/- 4.57, Hard 7.20 +/- 4.71; ratio 1.41x [0.84, 2.66]; t = 1.21,
  p = 0.24; Wilcoxon p = 0.31; d = 0.27; Hard > Medium 10/3/7.
- First 300 steps: Medium 41.95 +/- 10.84, Hard 27.80 +/- 7.53; ratio 0.66x [0.55, 0.80];
  t = -4.06, p = 6.7e-4; Wilcoxon p = 0.0011; d = -0.91; Hard > Medium 2/0/18.
- First 300 steps, deaths divided by alive count at phase start: Medium 0.58 +/- 0.09,
  Hard 0.65 +/- 0.10; ratio 1.12x [1.01, 1.23]; t = 2.15, p = 0.045; Wilcoxon p = 0.048;
  d = 0.48; 13/0/7.

### (c) No-shift baselines

Within the same runs:

- Medium, per-check deaths, epoch 1 vs epoch 0: 20.43 +/- 4.76 vs 5.10 +/- 4.57; p = 1.3e-11;
  epoch 1 higher in 20/20 seeds. (The Medium burst occurs in epoch 1.)
- Hard, per-check deaths, epoch 1 vs epoch 0: 4.05 +/- 2.11 vs 13.90 +/- 3.77; p = 3.5e-9;
  epoch 0 higher in 20/20 seeds.
- Medium, mid-phase 300 steps vs first 300 steps: 5.70 +/- 5.13 vs 41.95 +/- 10.84; ratio 7.36x;
  p = 4.0e-11; 20/20.
- Hard, mid-phase 300 steps vs first 300 steps: 10.00 +/- 6.28 vs 27.80 +/- 7.53; ratio 2.78x;
  p = 4.6e-8; 19/1/0.
- Transition excess (first 300 minus mid-phase 300), Medium vs Hard: 36.25 +/- 11.81 vs
  17.80 +/- 8.91; ratio 0.49x [0.35, 0.66]; t = -4.44, p = 2.8e-4; Wilcoxon p = 7.3e-4;
  d = -0.99; Hard > Medium 2/0/18.

Timing-matched no-shift control (separate runs, same 20 seeds). Easy classes are used in all
three phases, resampled to 30,000 and 50,000 samples per epoch in phases 2 and 3, so every phase
length and every check step is identical to the original curriculum, but the data distribution
never changes.

- Paper metric in the no-shift run: phase-2 slot 5.15 +/- 4.60, phase-3 slot 12.80 +/- 5.69;
  ratio 2.49x [1.67, 4.21]; t = 4.49, p = 2.5e-4; Wilcoxon p = 0.0010; d = 1.00; 16/0/4.
- Medium-slot epoch-0 deaths, no shift vs real 4->6 shift: 5.15 vs 5.10; ratio 0.99x; p = 0.58;
  identical in 17/20 seeds.
- Hard-slot epoch-0 deaths, no shift vs real 6->10 shift: 12.80 vs 27.80; ratio 2.17x [1.81, 2.67];
  p = 8.3e-8; Wilcoxon p = 1.2e-4; d = 1.87; 19/0/1.
- Excess deaths over the no-shift run, first 300 steps, 4->6 vs 6->10: 33.50 +/- 7.63 vs
  15.00 +/- 7.80; ratio 0.45x [0.34, 0.56]; t = -7.49, p = 4.4e-7; Wilcoxon p = 1.2e-4;
  d = -1.67; 6->10 larger in 1/0/19 seeds.
- Fraction of alive neurons eligible for death at the first fully post-shift check
  (step 1050 vs 2250): 0.51 +/- 0.06 vs 0.63 +/- 0.06; ratio 1.22x [1.14, 1.29]; t = 6.08,
  p = 7.6e-6; Wilcoxon p = 1.9e-5; d = 1.36; 18/0/2.
- Same fraction in the no-shift run: 0.05 +/- 0.05 vs 0.08 +/- 0.05; ratio 1.83x; p = 0.023.
- Same fraction, excess over the no-shift run: 0.47 +/- 0.06 vs 0.54 +/- 0.07; ratio 1.16x
  [1.08, 1.23]; t = 4.17, p = 5.2e-4; Wilcoxon p = 5.9e-4; d = 0.93; 16/0/4.

### (d) Control curricula (20 seeds each)

Equal-length curriculum (4 -> 6 -> 10 classes, every phase subsampled to 20,000 samples, i.e.
157 steps per epoch; exactly one check in each phase's epoch 0):

- Epoch-0 deaths: 4->6 5.40 +/- 3.35, 6->10 4.30 +/- 4.09; ratio 0.80x [0.49, 1.16]; t = -1.11,
  p = 0.28; Wilcoxon p = 0.30; d = -0.25; 6->10 larger in 7/2/11.
- First 300 steps: 40.95 +/- 8.15 vs 25.50 +/- 7.49; ratio 0.62x [0.53, 0.73]; t = -5.87,
  p = 1.2e-5; Wilcoxon p = 2.9e-4; d = -1.31; 2/1/17.

Reversed curriculum (10 -> 6 -> 4 classes; epochs 5, 5, 10 by position):

- Epoch-0 deaths: 10->6 6.95 +/- 3.12, 6->4 6.00 +/- 4.01; ratio 0.86x [0.66, 1.09]; t = -1.16,
  p = 0.26; Wilcoxon p = 0.26; d = -0.26; 5/3/12.
- First 300 steps: 11.15 +/- 4.08 vs 11.40 +/- 5.30; ratio 1.02x [0.82, 1.26]; p = 0.85;
  d = 0.04; 9/2/9.

Removing classes produced no death burst above the baseline rate.

Final 10-class test accuracy: original 75.7 +/- 1.3%; equal-length 72.2 +/- 1.4%;
reversed 44.3 +/- 2.7% (ends on 4 classes); no-shift 35.2 +/- 0.4% (sees only 4 classes).


## Part 3: Binary Per-pass Activation Rate

Method. The per-pass activation indicator was changed to binary: a neuron counts as active on a
forward pass if any sample in the batch has pre-activation > 0, averaged over the last 150 passes.
Everything else is unchanged. The original curriculum was run with the batch-1 seeds (n = 10).

- Epoch-0 deaths (paper metric): Medium 4.90 +/- 3.56, Hard 28.00 +/- 5.98; ratio 5.71x
  [3.74, 10.78]; t = 9.16, p = 7.4e-6; Wilcoxon p = 0.0020; d = 2.90; 10/0/0.
- Per check: 4.90 vs 14.00; ratio 2.86x [1.87, 5.39]; p = 4.6e-4; Wilcoxon p = 0.0020; d = 1.69;
  10/0/0.
- First 150 steps: 4.90 +/- 3.56 vs 4.10 +/- 2.91; ratio 0.84x [0.43, 1.76]; p = 0.63;
  Wilcoxon p = 0.55; d = -0.16; 4/1/5.
- First 300 steps: 41.20 +/- 7.90 vs 28.00 +/- 5.98; ratio 0.68x [0.54, 0.86]; p = 0.012;
  Wilcoxon p = 0.014; d = -0.99; 2/0/8.
- For comparison, the fractional rate on the same 10 seeds gives 4.47x [2.78, 8.11], p = 9.3e-5.
- Mean final accuracy with binary rate: 76.0%. Mean binary rate of alive neurons at checks: 0.73.

The burst effect holds under the paper's metric with the binary definition, and shows the same
dependence on the counting window. Deaths are dominated by neurons whose pre-activation is <= 0 for
every sample in nearly every batch, which are inactive under either definition.


## Interpretation for the Paper

Supported by the data:

- The experiment is exactly reproducible from the published code and seeds.
- Distribution shifts that add classes trigger a burst of neuron deaths, arriving at the second
  reorganize() check after the shift (when the 150-step rate window is fully post-shift).
- The 6->10 shift silences a moderately larger fraction of the surviving population than the
  4->6 shift (excess over a timing-matched no-shift baseline: 54% vs 47%, 1.16x, p = 5e-4, 16/20).

Not supported by the data:

- That burst magnitude in absolute deaths scales with shift severity. With equal-opportunity
  windows the easier transition produces more deaths (0.66x, 0.62x, 0.45x across three measures).
- The 5.45x ratio as a measure of shift sensitivity. About half of it (2.49x) appears with no
  shift at all, and the Easy->Medium count is unaffected by the shift (17/20 seeds identical to
  the no-shift run).
- That removing classes produces bursts (reversed curriculum: none).

Known confounds in the original measurement:

1. Unequal counting windows: epoch 0 is 235 steps for Medium and 391 for Hard, giving 1 vs 2 checks.
2. The Medium burst falls at step 1050, after Medium epoch 0 ends at step 1020.
3. Different alive populations at the two transitions (71 vs 42 neurons).
4. The N_min = 16 floor caps deaths during the Hard phase.


## Deviations and Limitations

- Hardware and software differ from (unrecorded) original versions; seed-level agreement was
  nevertheless exact.
- Control curricula use subsets or resamples drawn with a private random generator
  (seed 10000 * seed + phase index) so the global RNG stream (initialisation, shuffling, child
  noise) is unchanged. The equal-length curriculum uses a fixed subset per seed, so the Hard
  phase sees only 40% of its data. The no-shift curriculum repeats samples to reach the required
  phase lengths.
- Ratios with a near-zero denominator (e.g. excess deaths in the first 150 steps, mean -0.05)
  are not interpretable and are not reported.
- Not run: Part 3 with the batch-2 seeds (10 seeds were specified), datasets other than CIFAR-10
  (SVHN was run afterwards; see the addendum), additional seeds, or any hyperparameter changes.


## Files

```text
scripts/metnn_exp1.py     notebook code, verbatim
scripts/metnn_instr.py    instrumented copy, control curricula, binary rate
scripts/run_all.py        driver; scripts/analyze.py analysis; scripts/build_kernels.py Kaggle packaging
results/part1.txt         full output of the original protocol
results/part*_per_seed.csv, part*_per_check.csv, part*_per_epoch.csv   raw logs
results/analysis.md       all statistical tables
results/deaths_per_check_4panel.png   deaths at every check, four curricula (scripts/analyze.py)
results/svhn_*, results/svhn.txt, results/svhn_analysis.md     SVHN runs (addendum)
results/fisher_vs_rate*.csv, results/fisher_analysis.md        Fisher vs rate (addendum)
scripts/analyze_svhn_fisher.py, scripts/fisher_vs_rate.py      addendum analysis and runner
scripts/make_figures.py   paper figures
```

## Addendum: SVHN and Fisher-vs-Rate Reruns

These two analyses were run after the report above, on the same Kaggle Tesla T4 setup with the
same code. They are not part of the v2 paper. Full tables: results/svhn_analysis.md and
results/fisher_analysis.md.

SVHN (unchanged Experiment 1 code, same 20 seeds; instrumented runs matched it in 20/20 seeds).
The curriculum class indices are digits {0,1,8,9} -> {0,1,2,7,8,9} -> all ten. Because SVHN is
class-imbalanced, epochs are 223, 350 and 573 steps, and both counted first epochs contain their
first fully post-shift check (steps 1350 and 3150).

- Original metric: 4->6 42.55 +/- 5.42, 6->10 21.45 +/- 9.91; ratio 0.50x [0.40, 0.61];
  t = -7.91, p = 2.0e-7; Wilcoxon p = 1.0e-4; d = -1.77; 1/0/19. Reversed in each 10-seed batch
  separately (p = 0.0019 and 4.4e-5).
- First 300 steps: 42.55 +/- 5.42 vs 18.25 +/- 8.93; ratio 0.43x [0.34, 0.53]; 1/0/19.
- Eligible fraction at the first fully post-shift check: 0.64 vs 0.64; ratio 1.00x [0.94, 1.07];
  p = 0.99.
- Final 10-digit test accuracy: 91.07 +/- 0.31%.

This contradicts the earlier, unreleased SVHN report of no effect (p = 0.55, d = 0.20).

Fisher importance vs activation rate (Experiment 1 model at the end of the first phase, step 785;
diagonal empirical Fisher of the metabolic layer, summed per neuron; 10 seeds, 56-88 alive neurons
each):

- Spearman rho = +0.271 +/- 0.206 (mean +/- SD over seeds; range -0.232 to +0.448).
- One-sample t = 4.156, p = 0.00246; 9/10 seeds positive (sign test p = 0.0215).

This contradicts the earlier, unreleased figure of rho = 0.17 (not significant).
