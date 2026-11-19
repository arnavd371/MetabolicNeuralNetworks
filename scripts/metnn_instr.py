# =============================================================================
# Instrumented copy of metnn_exp1._run_curriculum (original file is untouched).
#
# The training loop below is a line-for-line copy of the original, with only
# read-only logging added. Logging consumes no RNG, so for the original
# curriculum + rate_mode="frac" the per-seed transition deaths must match
# metnn_exp1._run_curriculum exactly (checked in part2.py).
#
# Additions:
#   * per-check log: step, phase, epoch, deaths, births, alive before/after,
#     #alive neurons below theta_death, #steps of the current phase inside the
#     W=150 activation window, mean loss
#   * per-epoch log: start/end step, #steps, #checks, deaths, births
#   * alternative curricula (reversed, equal-length, matched no-shift)
#   * rate_mode="binary": per-pass indicator (neuron counts as active on a
#     pass if ANY sample in the batch has pre > 0)
#
# Data subsets for control curricula are drawn with a private torch.Generator
# so the global RNG stream (model init, shuffling, child noise) is not shifted.
# =============================================================================
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import metnn_exp1 as M

EASY, MED, HARD = [0, 1, 8, 9], [0, 1, 8, 9, 2, 7], list(range(10))

# (name, classes, epochs, n_samples_per_epoch or None = all samples of classes)
CURRICULA = {
    # identical to M.CURRICULUM
    "original":  [("Easy", EASY, 5, None), ("Medium", MED, 5, None),
                  ("Hard", HARD, 10, None)],
    # Hard-first: 10 -> 6 -> 4, epochs kept by position (5, 5, 10)
    "reversed":  [("Hard", HARD, 5, None), ("Medium", MED, 5, None),
                  ("Easy", EASY, 10, None)],
    # 4 -> 6 -> 10 with every phase subsampled to 20000 samples (157 steps/epoch)
    "equal_len": [("Easy", EASY, 5, 20000), ("Medium", MED, 5, 20000),
                  ("Hard", HARD, 10, 20000)],
    # No distribution shift at all, but phase lengths (and therefore every
    # reorganize() step) identical to the original: Easy classes throughout,
    # resampled to 30000 / 50000 samples per epoch in slots 2 and 3.
    "noshift_matched": [("Easy", EASY, 5, None), ("Medium", EASY, 5, 30000),
                        ("Hard", EASY, 10, 50000)],
}


class IdxLoader(M.FastLoader):
    """FastLoader over an explicit index multiset (same shuffling code path)."""
    def __init__(self, handle, idx, batch=128, shuffle=True):
        self.x, self.y = handle
        self.idx = idx
        self.batch, self.shuffle = batch, shuffle


def make_loader(train, classes, n, seed, phase_i):
    base = M.loader(train, classes)               # original code path
    if n is None:
        return base
    g = torch.Generator().manual_seed(10_000 * seed + phase_i)
    idx = base.idx.cpu()
    if n <= len(idx):                              # subsample without replacement
        sel = idx[torch.randperm(len(idx), generator=g)[:n]]
    else:                                          # every sample once + random extras
        extra = idx[torch.randint(0, len(idx), (n - len(idx),), generator=g)]
        sel = torch.cat([idx, extra])
    return IdxLoader(train, sel.to(base.idx.device))


def _binary_forward(self, x):
    pre = self.linear(x)
    if self.training:
        with torch.no_grad():
            f = (pre.float() > 0).any(0).float()
            self.act_history.append(f)
    return F.relu(pre) * self.alive.float()


def run_curriculum_logged(seed, curriculum="original", rate_mode="frac",
                          ds="cifar10", synthetic=False):
    spec = CURRICULA[curriculum]
    M.set_seed(seed)
    train, _ = M.get_dataset(ds, synthetic)
    model = M.MetNN(N=256).to(M.DEVICE)
    if rate_mode == "binary":
        model.metabolic.forward = _binary_forward.__get__(model.metabolic)
    elif rate_mode != "frac":
        raise ValueError(rate_mode)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=M.USE_AMP)
    crit  = nn.CrossEntropyLoss()
    step  = 0
    phase_alive, transition_deaths = {}, {}
    checks, epochs_log = [], []
    met = model.metabolic
    t0 = time.time()

    for pi, (name, classes, epochs, n) in enumerate(spec):
        ldr = make_loader(train, classes, n, seed, pi)
        phase_start = step + 1
        for ep in range(epochs):
            model.train()
            ep_deaths = ep_births = ep_checks = 0
            ep_start = step + 1
            losses = []
            for x, y in ldr:
                x, y = x.to(M.DEVICE), y.to(M.DEVICE)
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=M.USE_AMP):
                    loss = crit(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
                step += 1
                lv = loss.item()
                met.record_loss(lv)
                losses.append(lv)
                if step % met.C == 0:
                    # ---- read-only snapshot before the check ----
                    r = met.activation_rate()
                    alive_before = met.num_alive()
                    below = int(((r < met.theta_death) & met.alive).sum())
                    win = list(met.loss_history)
                    # ---- original call ----
                    d, b, _ = met.reorganize(opt)
                    ep_deaths += d
                    ep_births += b; ep_checks += 1
                    checks.append(dict(
                        curriculum=curriculum, rate_mode=rate_mode, seed=seed,
                        step=step, phase=name, phase_idx=pi, epoch=ep,
                        step_in_phase=step - phase_start + 1,
                        step_in_epoch=step - ep_start + 1,
                        window_steps_in_phase=min(met.W, step - phase_start + 1),
                        deaths=d, births=b, alive_before=alive_before,
                        alive_after=met.num_alive(), below_death_before=below,
                        mean_r_alive=float(r[met.alive].mean()) if met.num_alive() else 0.0,
                        loss_window_mean=sum(win) / max(len(win), 1),
                    ))
            if ep == 0:
                transition_deaths[name] = ep_deaths
            epochs_log.append(dict(
                curriculum=curriculum, rate_mode=rate_mode, seed=seed,
                phase=name, phase_idx=pi, epoch=ep, start_step=ep_start,
                end_step=step, n_steps=step - ep_start + 1, n_checks=ep_checks,
                deaths=ep_deaths, births=ep_births,
                mean_loss=sum(losses) / max(len(losses), 1),
                alive_end=met.num_alive()))
        phase_alive[name] = met.num_alive()

    _, test = M.get_dataset(ds, synthetic)
    acc = M.evaluate(model, M.loader(test, list(range(10)), train=False))
    return {"alive": phase_alive, "td": transition_deaths, "acc": acc,
            "checks": checks, "epochs": epochs_log, "total_steps": step,
            "seconds": time.time() - t0}
