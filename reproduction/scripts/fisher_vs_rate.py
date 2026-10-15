"""Fisher importance vs activation rate, per metabolic neuron.

Setting: the Experiment 1 model (MetNN, N=256, single head) trained on the
Experiment 1 curriculum, stopped at the end of the first phase (Easy, classes
0/1/8/9, 5 epochs = 785 steps). The training loop is metnn_exp1._run_curriculum
truncated after phase 0, so every seed's state at step 785 is identical to the
state in the Experiment 1 runs. Experiment 1 was chosen over Split-CIFAR-10
task 0 because Experiment 1 is the model whose activation rates the paper
analyses; Experiment 2 uses a different multi-head model with N_min=64.

r_i:  model.metabolic.activation_rate() at step 785 (mean of the last W=150
      per-step firing fractions, exactly what reorganize() uses).
F_i:  Experiment 2's _fisher definition (eval mode, fp32, mean over 8 batches
      of the squared gradient of the batch loss), restricted to the metabolic
      layer and summed per neuron over its incoming weight row and bias.
      Batches: the first 8 batches of a fresh shuffled pass over phase-0 data.

Usage: python fisher_vs_rate.py --out results [--seeds ...]
"""
import argparse, csv, os
import numpy as np
import torch
import torch.nn as nn
from scipy import stats as st

import metnn_exp1 as M


def train_phase0(seed, ds="cifar10"):
    M.set_seed(seed)
    train, _ = M.get_dataset(ds)
    model = M.MetNN(N=256).to(M.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=M.USE_AMP)
    crit = nn.CrossEntropyLoss()
    step = 0
    name, classes, epochs = M.CURRICULUM[0]
    ldr = M.loader(train, classes)
    for ep in range(epochs):                      # identical to _run_curriculum
        model.train()
        for x, y in ldr:
            x, y = x.to(M.DEVICE), y.to(M.DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=M.USE_AMP):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            step += 1
            model.metabolic.record_loss(loss.item())
            if step % model.metabolic.C == 0:
                model.metabolic.reorganize(opt)
    return model, ldr, step


def neuron_fisher(model, ldr, n_batches=8):
    crit = nn.CrossEntropyLoss()
    lin = model.metabolic.linear
    Fw = torch.zeros_like(lin.weight); Fb = torch.zeros_like(lin.bias)
    model.eval(); done = 0
    for i, (x, y) in enumerate(ldr):
        if i >= n_batches: break
        model.zero_grad()
        crit(model(x), y).backward()
        Fw += lin.weight.grad.detach() ** 2; Fb += lin.bias.grad.detach() ** 2
        done += 1
    return ((Fw.sum(1) + Fb) / max(done, 1)).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123, 0, 99, 11, 22, 33, 44, 55])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rows, neuron_rows = [], []
    for s in a.seeds:
        model, ldr, step = train_phase0(s)
        met = model.metabolic
        r = met.activation_rate().float().cpu()
        alive = met.alive.cpu()
        F_ = neuron_fisher(model, ldr)
        ri, fi = r[alive].numpy(), F_[alive].numpy()
        rho, p = st.spearmanr(ri, fi)
        rows.append(dict(seed=s, step=step, n_alive=int(alive.sum()), spearman_rho=float(rho),
                         p=float(p), r_mean=float(ri.mean()), fisher_median=float(np.median(fi))))
        for idx in alive.nonzero().flatten().tolist():
            neuron_rows.append(dict(seed=s, neuron=idx, rate=float(r[idx]), fisher=float(F_[idx])))
        print(f"seed {s:3d}  alive {int(alive.sum()):3d}  rho={rho:+.3f}  p={p:.3g}", flush=True)
    for name, rr in (("fisher_vs_rate.csv", rows), ("fisher_vs_rate_per_neuron.csv", neuron_rows)):
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
    rhos = np.array([x["spearman_rho"] for x in rows])
    t, p = st.ttest_1samp(rhos, 0.0)
    sign = st.binomtest(int((rhos > 0).sum()), len(rhos), 0.5).pvalue
    print(f"rho mean {rhos.mean():+.3f}  SD {rhos.std(ddof=1):.3f}  min {rhos.min():+.3f}  "
          f"max {rhos.max():+.3f}  one-sample t={t:.3f} p={p:.3g}  positive {int((rhos>0).sum())}/{len(rhos)} "
          f"sign-test p={sign:.3g}")


if __name__ == "__main__":
    main()
