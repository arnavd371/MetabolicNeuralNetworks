# Metabolic Neural Networks

**One local signal, three structural decisions: how neurons can grow, prune, and protect themselves without being told to.**

This repository contains the implementation and experimental results for a project testing whether a single local statistic, a neuron's rolling activation rate, can drive structural decisions during neural network training, removing unused neurons, growing overloaded ones, and protecting important ones, with no global supervision, fixed schedules, or external task labels.

## The idea

Each neuron tracks one private number: `r(i)`, the fraction of recent forward passes on which it fired. Three fixed thresholds turn this single number into structural decisions:

| Decision | Condition | Effect |
|---|---|---|
| **Death** | `r(i) < 0.05` for 150 steps | Neuron is removed |
| **Birth** | `r(i) > 0.70` during a loss plateau | Neuron is duplicated, with noise |
| **Consolidation** | `r(i) > 0.30` after a burst of 5+ deaths in one check | Neuron is frozen against further updates |

No part of this mechanism is told when a task changes or how hard it is. All decisions are inferred purely from each neuron's own local activation history.

## What's in this repo

- `notebooka3b8ccac17.ipynb` — Kaggle notebook implementing the `MetabolicLayer` and Experiment 1: **Shift Detection via Reorganization Bursts**.

## Experiment 1: Does the network notice when tasks get harder?

A 3-phase CIFAR-10 curriculum (4 classes, then 6, then all 10) was run with no signal given to the network about phase transitions. Neuron deaths in the first epoch following each transition were recorded across two independent batches of 10 random seeds each.

| Batch | Easy→Medium deaths | Medium→Hard deaths | Ratio | p-value | Cohen's d |
|---|---|---|---|---|---|
| 1 (seeds 42–55) | 6.0 ± 4.2 | 26.8 ± 7.1 | 4.47× | 0.0001 | 2.10 |
| 2 (seeds 100–109) | 4.2 ± 4.7 | 28.8 ± 7.8 | 6.86× | 0.0001 | 2.17 |
| **Pooled (n=20)** | **5.1 ± 4.4** | **27.8 ± 7.4** | **5.45×** | **<0.0001** | **2.15** |

Combining both batches with Stouffer's method gives an overall p ≈ 0.00000, with the harder transition producing larger reorganization bursts in 20 out of 20 seeds tested. This result replicated independently across two batches run at different times.

Full statistical battery included: paired t-test, Wilcoxon signed-rank test, Cohen's d, bootstrap 95% confidence intervals on the ratio, Shapiro-Wilk normality check, and a formal meta-analysis combining both batches.

## Experiment 2: Does protecting memory reduce catastrophic forgetting?

A second experiment, comparing a static baseline, Elastic Weight Consolidation (EWC), and the metabolic mechanism with and without a consolidation/protection step, on Split-CIFAR-10 continual learning, was also run as part of this project. Its code is not included in this notebook; full results are reported in the accompanying written research report:

| Method | Final accuracy | Forgetting |
|---|---|---|
| Static baseline | 76.25% | 16.83% |
| EWC | 83.13% | 7.07% |
| Metabolic, no protection | 74.59% | 18.89% |
| Metabolic, with protection | 76.16% | 16.13% |

Unprotected structural plasticity increased forgetting relative to the static baseline; adding the protection step recovered most of that gap, though the improvement did not reach statistical significance at the sample sizes tested (p = 0.27–0.37 across runs).

## Usage

In a fresh notebook cell, paste the contents of `notebooka3b8ccac17.ipynb`'s code cell, then:

```python
run("smoke")                                              # 30s sanity check
run("exp1", seeds=[42,7,123,0,99,11,22,33,44,55], batch_name="batch1")
run("exp1", seeds=[100,101,102,103,104,105,106,107,108,109], batch_name="batch2")
run("meta")                                                # combine both batches
run("figures")                                             # save result PNGs
run("summary")                                              # paper-ready table
```

All experiments use deterministic GPU settings to ensure exact reproducibility from a given seed.

## Status

Experiment 1 is fully reproduced and verified in this repository, including raw per-seed output. Experiment 2's code and a fresh reproducible run are planned as a follow-up addition.

## Citation

If you use this work, please cite the accompanying research report: *One Local Signal, Three Structural Decisions: How Neurons Can Grow, Prune, and Protect Themselves Without Being Told To*, Dhiman & Makhambetaliyeva, 2026.
