# =============================================================================
# MetNN — Exp 1 Only: Shift Detection via Reorganization Bursts
# "One local signal, three structural decisions" (death/birth/consolidation
# all live in the layer below; this file's EXPERIMENT only tests the
# detection claim: does burst magnitude scale with distributional shift?)
#
# Speed features kept from the fast version:
#   • AMP (mixed precision)        • GPU-resident cached datasets
#   • BatchNorm backbone            • deterministic seeding (reproducible)
#
# Statistics now reported (the additions you asked for):
#   • Paired t-test + Wilcoxon signed-rank + Cohen's d        (per batch)
#   • Pooled / meta-analytic test combining multiple seed batches
#   • Bootstrap 95% CI on the ratio statistic
#   • Per-seed monotonicity check (consistency rate)
#   • Shapiro-Wilk normality check on the paired differences
#     (justifies whether the t-test or the Wilcoxon is the one to trust)
#
# Usage in Colab (paste whole file, then):
#   run("smoke")                              # 30s sanity check
#   run("exp1", seeds=[...])                  # one batch
#   run("exp1", seeds=[...], datasets=("cifar10","svhn"))
#   run("meta")                                # combine ALL exp1 batches run so far
#   run("figures")                             # save PNGs
#   run("summary")                             # paper-ready table
# =============================================================================

from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as sp_stats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")
RESULTS = {}
print(f"Device: {DEVICE}  AMP: {USE_AMP}")

if DEVICE == "cuda":
    # benchmark=True is non-deterministic (autotuned conv algorithms can
    # differ run to run). Since this layer makes structural decisions from
    # tiny floating-point differences in activation rates, that nondeterminism
    # silently changes which neurons die/divide. deterministic=True trades a
    # little speed for results that are exactly reproducible from a seed.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_seed(seed):
    """Full deterministic seeding — covers CPU, CUDA, and numpy RNGs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ── MetabolicLayer (death + birth only are exercised by Exp 1; the gradient
#    masking / consolidation machinery is kept so the layer's interface stays
#    consistent with the rest of the MetNN line, but Exp 1 never triggers it
#    since consolidate_on_burst defaults off here) ───────────────────────────
class MetabolicLayer(nn.Module):
    """
    Local per-neuron activation rate r(i) drives two structural decisions
    exercised in this experiment:
      DEATH   r(i) < θ_death              → neuron removed
      BIRTH   r(i) > θ_birth + plateau    → neuron divides
    (Consolidation logic is present but inactive by default; Exp 1 only
    measures detection via death bursts, not protection.)
    """
    def __init__(self, in_features, N=256, init_alive=None,
                 theta_death=0.05, theta_birth=0.70, theta_protect=0.30,
                 W=150, C=150, N_min=16, B_max=10,
                 k_plateau=10, eps_plateau_rel=0.01,
                 child_noise=0.015, warmup_checks=3,
                 burst_threshold=5, consolidate_on_burst=False,
                 mode="full"):
        super().__init__()
        self.N = N
        self.linear = nn.Linear(in_features, N)
        if init_alive is None:
            init_alive = N // 2
        alive = torch.zeros(N, dtype=torch.bool)
        alive[:init_alive] = True
        self.register_buffer("alive", alive)
        self.register_buffer("consolidated", torch.zeros(N, dtype=torch.bool))
        with torch.no_grad():
            self.linear.weight[~alive] = 0.0
            self.linear.bias[~alive]   = 0.0
        self.linear.weight.register_hook(self._mask_grad)
        self.linear.bias.register_hook(self._mask_grad)
        self.theta_death   = theta_death
        self.theta_birth   = theta_birth
        self.theta_protect = theta_protect
        self.W = W; self.C = C
        self.N_min = N_min; self.B_max = B_max
        self.k = k_plateau; self.eps_rel = eps_plateau_rel
        self.child_noise = child_noise
        self.warmup_checks = warmup_checks
        self.burst_threshold = burst_threshold
        self.consolidate_on_burst = consolidate_on_burst
        self.mode = mode
        self.act_history  = deque(maxlen=W)
        self.loss_history = deque(maxlen=2 * k_plateau)
        self.check_count  = 0

    def _mask_grad(self, grad):
        if self.consolidated.any():
            grad = grad.clone()
            grad[self.consolidated] = 0.0
        return grad

    def forward(self, x):
        pre = self.linear(x)
        if self.training:
            with torch.no_grad():
                f = (pre.float() > 0).float().mean(0)
                self.act_history.append(f)
        return F.relu(pre) * self.alive.float()

    def activation_rate(self):
        if not self.act_history:
            return torch.zeros(self.N, device=self.linear.weight.device)
        return torch.stack(list(self.act_history)).mean(0)

    def record_loss(self, v):
        self.loss_history.append(float(v))

    def in_plateau(self):
        if len(self.loss_history) < 2 * self.k:
            return False
        h = list(self.loss_history)
        first  = sum(h[:self.k]) / self.k
        second = sum(h[self.k:]) / self.k
        return (first - second) / (abs(first) + 1e-8) < self.eps_rel

    def num_alive(self):        return int(self.alive.sum())
    def num_consolidated(self): return int(self.consolidated.sum())

    @torch.no_grad()
    def reorganize(self, opt=None):
        self.check_count += 1
        r = self.activation_rate()
        nd = nb = nc = 0

        if self.mode in ("full", "death_only", "random"):
            alive_idx = self.alive.nonzero(as_tuple=True)[0]
            alive_idx = alive_idx[~self.consolidated[alive_idx]]
            victims = (alive_idx[torch.rand(len(alive_idx)) < 0.03]
                       if self.mode == "random"
                       else alive_idx[r[alive_idx] < self.theta_death])
            for i in victims.tolist():
                if self.num_alive() <= self.N_min: break
                self._kill(i, opt); nd += 1

        if (self.consolidate_on_burst
                and self.mode not in ("none", "random")
                and nd >= self.burst_threshold
                and self.check_count > self.warmup_checks):
            nc = self._consolidate()

        if (self.mode in ("full", "birth_only", "random")
                and self.check_count > self.warmup_checks):
            plateau = True if self.mode == "random" else self.in_plateau()
            if plateau:
                dead = (~self.alive).nonzero(as_tuple=True)[0].tolist()
                aidx = self.alive.nonzero(as_tuple=True)[0]
                if self.mode == "random":
                    perm  = torch.randperm(len(aidx))
                    prnts = aidx[perm][:self.B_max].tolist()
                else:
                    over  = aidx[r[aidx] > self.theta_birth]
                    over  = over[torch.argsort(r[over], descending=True)]
                    prnts = over[:self.B_max].tolist()
                for p in prnts:
                    if not dead: break
                    self._spawn(p, dead.pop(), opt); nb += 1
        return nd, nb, nc

    @torch.no_grad()
    def _consolidate(self, min_rate=None):
        r = self.activation_rate()
        t = self.theta_protect if min_rate is None else min_rate
        m = self.alive & (r > t) & (~self.consolidated)
        self.consolidated |= m
        return int(m.sum())

    @torch.no_grad()
    def _kill(self, i, opt):
        self.alive[i] = False
        self.linear.weight[i] = 0.0; self.linear.bias[i] = 0.0
        self._zero_opt(opt, i)

    @torch.no_grad()
    def _spawn(self, parent, slot, opt):
        self.linear.weight[slot] = (self.linear.weight[parent]
            + torch.randn_like(self.linear.weight[parent]) * self.child_noise)
        self.linear.bias[slot] = self.linear.bias[parent].clone()
        self.alive[slot] = True
        self._zero_opt(opt, slot)

    @torch.no_grad()
    def _zero_opt(self, opt, idx):
        if opt is None: return
        for g in opt.param_groups:
            for p in g["params"]:
                if p is self.linear.weight or p is self.linear.bias:
                    for k in ("exp_avg", "exp_avg_sq"):
                        if k in opt.state.get(p, {}):
                            opt.state[p][k][idx] = 0.0


# ── Model ─────────────────────────────────────────────────────────────────────
class Backbone(nn.Module):
    """CNN with BatchNorm — converges faster, needs fewer epochs."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
    def forward(self, x): return self.net(x).flatten(1)


class MetNN(nn.Module):
    def __init__(self, n_classes=10, N=256, **kw):
        super().__init__()
        self.backbone  = Backbone()
        self.metabolic = MetabolicLayer(2048, N=N, **kw)
        self.head      = nn.Linear(N, n_classes)
    def forward(self, x):
        return self.head(self.metabolic(self.backbone(x)))


# ── Data (GPU-resident, cached) ───────────────────────────────────────────────
_CACHE = {}
_STATS = {
    "cifar10": ((0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)),
    "svhn":    ((0.4377,0.4438,0.4728), (0.1980,0.2010,0.1970)),
}


def get_dataset(name="cifar10", synthetic=False):
    """Return (train_handle, test_handle); each handle is a (x, y) GPU tensor pair."""
    if synthetic:
        g = torch.Generator().manual_seed(0)
        xtr = torch.randn(2000, 3, 32, 32, generator=g)
        ytr = torch.randint(0, 10, (2000,), generator=g)
        xte = torch.randn(500, 3, 32, 32, generator=g)
        yte = torch.randint(0, 10, (500,), generator=g)
        return (xtr.to(DEVICE), ytr.to(DEVICE)), (xte.to(DEVICE), yte.to(DEVICE))

    if name in _CACHE:
        return _CACHE[name]

    from torchvision import datasets
    mean, std = _STATS[name]
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std  = torch.tensor(std).view(1, 3, 1, 1)

    def normalize(x_uint8):
        x = x_uint8.float().div_(255.0)
        return (x - mean) / std

    if name == "cifar10":
        tr = datasets.CIFAR10("./data", train=True,  download=True)
        te = datasets.CIFAR10("./data", train=False, download=True)
        xtr = normalize(torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous())
        xte = normalize(torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous())
        ytr = torch.tensor(tr.targets); yte = torch.tensor(te.targets)
    elif name == "svhn":
        tr = datasets.SVHN("./data", split="train", download=True)
        te = datasets.SVHN("./data", split="test",  download=True)
        xtr = normalize(torch.from_numpy(tr.data).contiguous())
        xte = normalize(torch.from_numpy(te.data).contiguous())
        ytr = torch.tensor(tr.labels); yte = torch.tensor(te.labels)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    handle = ((xtr.to(DEVICE), ytr.to(DEVICE)),
              (xte.to(DEVICE), yte.to(DEVICE)))
    _CACHE[name] = handle
    return handle


class FastLoader:
    """Iterates GPU-resident minibatches for a fixed set of classes."""
    def __init__(self, handle, classes, batch=128, shuffle=True):
        x, y = handle
        mask = torch.zeros(len(y), dtype=torch.bool, device=y.device)
        for c in classes:
            mask |= (y == c)
        self.idx = mask.nonzero(as_tuple=True)[0]
        self.x, self.y = x, y
        self.batch, self.shuffle = batch, shuffle

    def __iter__(self):
        idx = self.idx
        if self.shuffle:
            idx = idx[torch.randperm(len(idx), device=idx.device)]
        for i in range(0, len(idx), self.batch):
            b = idx[i:i + self.batch]
            yield self.x[b], self.y[b]


def loader(dataset, classes, batch=128, synthetic=False, n_fake=1000, train=True):
    return FastLoader(dataset, classes, batch=batch, shuffle=train)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, ldr):
    model.eval()
    correct = total = 0
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=USE_AMP):
        for x, y in ldr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item(); total += y.numel()
    return 100.0 * correct / max(total, 1)


# ── Statistics ────────────────────────────────────────────────────────────────
def paired_tests(easy, hard):
    """
    Full statistical battery for the paired easy-vs-hard burst comparison.
    Returns a dict so callers/printers don't have to remember field order.
    """
    easy, hard = np.array(easy, float), np.array(hard, float)
    diff = hard - easy

    t, p = sp_stats.ttest_rel(hard, easy)
    try:
        _, wp = sp_stats.wilcoxon(hard, easy)
    except ValueError:               # all-zero differences edge case
        wp = 1.0
    d = diff.mean() / (diff.std(ddof=1) + 1e-12)

    # Shapiro-Wilk on the paired differences: tells you whether the t-test's
    # normality assumption is reasonable, or whether to lean on Wilcoxon.
    if len(diff) >= 3 and np.std(diff) > 0:
        sw_stat, sw_p = sp_stats.shapiro(diff)
    else:
        sw_stat, sw_p = float("nan"), float("nan")

    return dict(t=float(t), p=float(p), wilcoxon_p=float(wp), d=float(d),
                shapiro_stat=float(sw_stat), shapiro_p=float(sw_p),
                n=len(easy), mean_diff=float(diff.mean()),
                std_diff=float(diff.std(ddof=1)) if len(diff) > 1 else 0.0)


def bootstrap_ratio_ci(easy, hard, n_boot=10000, seed=0):
    """
    Bootstrap 95% CI on the ratio (mean hard-burst / mean easy-burst).
    Resamples seed-pairs with replacement, which respects the paired
    structure of the data (each resample keeps easy[i]/hard[i] coupled).
    """
    rng = np.random.default_rng(seed)
    easy, hard = np.array(easy, float), np.array(hard, float)
    n = len(easy)
    ratios = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        e_mean = easy[idx].mean() + 1e-6
        h_mean = hard[idx].mean()
        ratios[b] = h_mean / e_mean
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return float(lo), float(hi), float(np.median(ratios))


def monotonicity(easy, hard):
    """Fraction of seeds where hard > easy (the basic consistency check)."""
    easy, hard = np.array(easy), np.array(hard)
    consistent = int((hard > easy).sum())
    return consistent, len(easy)


def meta_analysis(batches):
    """
    Combine multiple independent seed batches into one pooled test using
    Stouffer's method on z-scores (standard fixed-effect meta-analysis for
    combining p-values from independent paired tests), weighted by sqrt(n).
    This is the right way to combine batch 1 (n=10, p=0.0010) and batch 2
    (n=10, p=0.0298) into a single overall significance statement, rather
    than reporting two p-values side by side and leaving the reader to guess
    how convinced to be.
    """
    zs, weights, ns = [], [], []
    pooled_easy, pooled_hard = [], []
    for b in batches:
        easy, hard = b["easy"], b["hard"]
        pooled_easy.extend(easy); pooled_hard.extend(hard)
        r = paired_tests(easy, hard)
        # one-sided z from the two-sided t-test p-value, signed by direction
        z = sp_stats.norm.isf(r["p"] / 2.0)
        if r["mean_diff"] < 0:
            z = -z
        zs.append(z)
        weights.append(np.sqrt(r["n"]))
        ns.append(r["n"])

    zs = np.array(zs); weights = np.array(weights)
    z_combined = (weights * zs).sum() / np.sqrt((weights ** 2).sum())
    p_combined = 2 * (1 - sp_stats.norm.cdf(abs(z_combined)))

    # also report the naive pooled paired test (treats all seeds as one
    # sample; valid here since each batch used the SAME curriculum protocol)
    pooled = paired_tests(pooled_easy, pooled_hard)

    return dict(z_combined=float(z_combined), p_combined=float(p_combined),
                n_batches=len(batches), total_n=sum(ns),
                pooled_paired=pooled,
                pooled_easy=pooled_easy, pooled_hard=pooled_hard)


# ── Curriculum setup ──────────────────────────────────────────────────────────
CURRICULUM = [
    ("Easy",   [0,1,8,9],          5),
    ("Medium", [0,1,8,9,2,7],      5),
    ("Hard",   list(range(10)),   10),
]


def _run_curriculum(seed, synthetic=False, ds="cifar10", n_fake=1000):
    set_seed(seed)
    train, _ = get_dataset(ds, synthetic)
    model = MetNN(N=256).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    crit  = nn.CrossEntropyLoss()
    step  = 0
    phase_alive, transition_deaths = {}, {}

    for name, classes, epochs in CURRICULUM:
        ldr = loader(train, classes, synthetic=synthetic, n_fake=n_fake)
        for ep in range(epochs):
            model.train()
            ep_deaths = 0
            for x, y in ldr:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    loss = crit(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
                step += 1
                model.metabolic.record_loss(loss.item())
                if step % model.metabolic.C == 0:
                    d, _, _ = model.metabolic.reorganize(opt)
                    ep_deaths += d
            if ep == 0:
                transition_deaths[name] = ep_deaths
        phase_alive[name] = model.metabolic.num_alive()

    _, test = get_dataset(ds, synthetic)
    acc = evaluate(model, loader(test, list(range(10)),
                                 synthetic=synthetic, n_fake=n_fake, train=False))
    return {"alive": phase_alive, "td": transition_deaths, "acc": acc}


def exp1(seeds, synthetic=False, ds="cifar10", batch_name=None):
    """
    Shift-detection experiment with the expanded statistics battery.
    Each call is treated as one independent batch and stored separately
    (keyed by dataset AND batch_name) so run("meta") can combine them later.
    """
    label = f"{ds}" + (f" / {batch_name}" if batch_name else "")
    print(f"\n── Exp 1: Shift Detection [{label}] ─────────────────────────")
    burst_easy, burst_hard, accs = [], [], []
    for s in seeds:
        r = _run_curriculum(s, synthetic=synthetic, ds=ds)
        burst_easy.append(r["td"]["Medium"])
        burst_hard.append(r["td"]["Hard"])
        accs.append(r["acc"])
        print(f"  seed {s:3d}  easy→med {burst_easy[-1]:4d}  "
              f"med→hard {burst_hard[-1]:4d}  acc {accs[-1]:.1f}%")

    be, bh = np.mean(burst_easy), np.mean(burst_hard)
    ratio  = bh / (be + 1e-6)
    consistent, n_seeds = monotonicity(burst_easy, burst_hard)
    r = paired_tests(burst_easy, burst_hard)
    ci_lo, ci_hi, ci_med = bootstrap_ratio_ci(burst_easy, burst_hard)

    print(f"\n  Easy→Medium  {be:.1f} ± {np.std(burst_easy):.1f}")
    print(f"  Medium→Hard  {bh:.1f} ± {np.std(burst_hard):.1f}")
    print(f"  Ratio        {ratio:.2f}×   bootstrap 95% CI [{ci_lo:.2f}, {ci_hi:.2f}]"
          f"  (median {ci_med:.2f})")
    print(f"  Consistent   {consistent}/{n_seeds} seeds (hard > easy)")
    print(f"  Paired t     t={r['t']:.3f}  p={r['p']:.4f}")
    print(f"  Wilcoxon     p={r['wilcoxon_p']:.4f}")
    print(f"  Cohen's d    {r['d']:.3f}")
    print(f"  Shapiro-Wilk on differences: p={r['shapiro_p']:.4f}  "
          f"({'normal-ish, trust t-test' if r['shapiro_p']>0.05 else 'non-normal, lean on Wilcoxon'})")
    print(f"  Accuracy     {np.mean(accs):.2f} ± {np.std(accs):.2f}%\n")

    key = f"exp1_{ds}" + (f"_{batch_name}" if batch_name else "")
    RESULTS[key] = dict(easy=burst_easy, hard=burst_hard, ds=ds,
                        batch_name=batch_name, ratio=ratio,
                        ci_lo=ci_lo, ci_hi=ci_hi, ci_med=ci_med,
                        consistent=consistent, n_seeds=n_seeds,
                        seeds=list(seeds), accs=accs, **r)
    return RESULTS[key]


def exp_meta(ds=None):
    """
    Combine ALL exp1 batches stored in RESULTS (optionally filtered to one
    dataset) into a single meta-analytic statement. Run this after calling
    exp1 multiple times with different seed sets, e.g. two independent
    10-seed batches — the standard way to ask "what's our overall confidence
    once both runs are considered together," rather than reporting two
    p-values side by side.
    """
    batches = [v for k, v in RESULTS.items()
               if k.startswith("exp1_") and (ds is None or v["ds"] == ds)]
    if len(batches) < 2:
        print(f"Need at least 2 stored exp1 batches to run meta-analysis "
              f"(found {len(batches)}). Run exp1 at least twice first, "
              f"e.g. with two different seed lists.")
        return None

    target_ds = ds or batches[0]["ds"]
    print(f"\n── Meta-analysis across {len(batches)} batches [{target_ds}] ──────")
    for b in batches:
        name = b["batch_name"] or "unnamed"
        print(f"  batch '{name}': n={b['n']}  p={b['p']:.4f}  d={b['d']:.3f}  "
              f"ratio={b['ratio']:.2f}×")

    m = meta_analysis(batches)
    pooled = m["pooled_paired"]
    pooled_ratio = (np.mean(m["pooled_hard"]) /
                    (np.mean(m["pooled_easy"]) + 1e-6))
    ci_lo, ci_hi, ci_med = bootstrap_ratio_ci(m["pooled_easy"], m["pooled_hard"])

    print(f"\n  Stouffer combined z = {m['z_combined']:.3f}, "
          f"p_combined = {m['p_combined']:.5f}  (total n={m['total_n']})")
    print(f"  Pooled paired test (all {m['total_n']} seeds as one sample):")
    print(f"    t={pooled['t']:.3f}  p={pooled['p']:.4f}  "
          f"Wilcoxon p={pooled['wilcoxon_p']:.4f}  d={pooled['d']:.3f}")
    print(f"  Pooled ratio = {pooled_ratio:.2f}×  "
          f"bootstrap 95% CI [{ci_lo:.2f}, {ci_hi:.2f}]")
    print(f"  This is the number to lead with in the paper: it reflects all "
          f"{m['total_n']} seeds across {m['n_batches']} independent runs.\n")

    RESULTS[f"meta_{target_ds}"] = dict(
        z_combined=m["z_combined"], p_combined=m["p_combined"],
        pooled_t=pooled["t"], pooled_p=pooled["p"],
        pooled_wilcoxon_p=pooled["wilcoxon_p"], pooled_d=pooled["d"],
        pooled_ratio=pooled_ratio, ci_lo=ci_lo, ci_hi=ci_hi,
        total_n=m["total_n"], n_batches=m["n_batches"], ds=target_ds)
    return RESULTS[f"meta_{target_ds}"]


# ── Figures ───────────────────────────────────────────────────────────────────
def exp_figures(save_dir="./figures"):
    import os, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    saved = []

    for key, r in RESULTS.items():
        if not key.startswith("exp1_"):
            continue
        tag = key[len("exp1_"):]
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        labels = ["Easy→Medium", "Medium→Hard"]
        means  = [np.mean(r["easy"]), np.mean(r["hard"])]
        stds   = [np.std(r["easy"]),  np.std(r["hard"])]
        ax.bar(labels, means, yerr=stds, capsize=5, width=0.5,
               color=["#9FE1CB", "#D85A30"], edgecolor="none")
        for i, vals in enumerate([r["easy"], r["hard"]]):
            ax.scatter([i] * len(vals), vals, c="#2C2C2A", s=18, zorder=5)
        ax.set_ylabel("Neuron deaths (first epoch of phase)", fontsize=10)
        title_bits = [f"ratio {r['ratio']:.2f}×", f"p={r['p']:.4f}",
                      f"d={r['d']:.2f}", f"{r['consistent']}/{r['n_seeds']} consistent"]
        ax.set_title(f"Reorganization burst magnitude [{tag}]\n" +
                     ", ".join(title_bits), fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = f"{save_dir}/fig1_bursts_{tag}.png"
        fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
        saved.append(p)

    if saved:
        print("Figures saved:")
        for p in saved: print(f"  {p}")
    else:
        print("No exp1 results yet. Run exp1 first.")
    return saved


# ── Summary ───────────────────────────────────────────────────────────────────
def exp_summary():
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY — Exp 1: Shift Detection via Reorganization Bursts")
    print("=" * 70)
    for key in sorted(RESULTS):
        r = RESULTS[key]
        if key.startswith("exp1_"):
            tag = key[len("exp1_"):]
            print(f"\n[{tag}]  n={r['n']} seeds")
            print(f"  Easy→Med {np.mean(r['easy']):.1f}±{np.std(r['easy']):.1f}   "
                  f"Med→Hard {np.mean(r['hard']):.1f}±{np.std(r['hard']):.1f}")
            print(f"  ratio {r['ratio']:.2f}×  CI[{r['ci_lo']:.2f},{r['ci_hi']:.2f}]  "
                  f"consistent {r['consistent']}/{r['n_seeds']}")
            print(f"  t-test p={r['p']:.4f}  Wilcoxon p={r['wilcoxon_p']:.4f}  "
                  f"d={r['d']:.3f}  Shapiro p={r['shapiro_p']:.4f}")
        elif key.startswith("meta_"):
            ds = key[len("meta_"):]
            print(f"\n[META across batches | {ds}]  total n={r['total_n']} "
                  f"({r['n_batches']} batches)")
            print(f"  Stouffer combined p = {r['p_combined']:.5f}")
            print(f"  Pooled paired test: p={r['pooled_p']:.4f}  "
                  f"Wilcoxon p={r['pooled_wilcoxon_p']:.4f}  d={r['pooled_d']:.3f}")
            print(f"  Pooled ratio {r['pooled_ratio']:.2f}×  "
                  f"CI[{r['ci_lo']:.2f},{r['ci_hi']:.2f}]")
    print("\n" + "=" * 70)


# ── Smoke test ────────────────────────────────────────────────────────────────
def exp_smoke():
    print("── Smoke test (synthetic, tiny) ─────────────────────────────")
    model = MetNN(N=32, C=3, W=3, warmup_checks=1, B_max=3,
                  k_plateau=2, N_min=4).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    crit = nn.CrossEntropyLoss()
    nd_total = nb_total = 0
    x = torch.randn(64, 3, 32, 32).to(DEVICE)
    y = torch.randint(0, 10, (64,)).to(DEVICE)
    for step in range(20):
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            loss = crit(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        model.metabolic.record_loss(loss.item())
        if (step + 1) % model.metabolic.C == 0:
            d, b, _ = model.metabolic.reorganize(opt)
            nd_total += d; nb_total += b
    print(f"  deaths {nd_total}  births {nb_total}  "
          f"alive {model.metabolic.num_alive()}")
    print(f"  AMP {'ON' if USE_AMP else 'OFF (CPU mode, still correct)'}")

    # quick check that the new stats functions don't error on toy data
    easy_toy = [2, 0, 3, 1, 4, 2, 0, 5, 1, 3]
    hard_toy = [8, 6, 9, 7, 5, 8, 6, 10, 7, 9]
    r = paired_tests(easy_toy, hard_toy)
    lo, hi, med = bootstrap_ratio_ci(easy_toy, hard_toy, n_boot=500)
    print(f"  stats self-test: p={r['p']:.4f}  d={r['d']:.2f}  "
          f"bootstrap CI=[{lo:.2f},{hi:.2f}]")
    print("  OK — all code paths executed.\n")


# ── Dispatcher ────────────────────────────────────────────────────────────────
def _in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def run(cmd, seeds=None, synthetic=False, datasets=("cifar10",),
        batch_name=None, ds=None):
    """
    Commands
    ────────
    run("smoke")                                    — 30s sanity check
    run("exp1", seeds=SEEDS, batch_name="batch1")   — run one batch
    run("exp1", seeds=SEEDS2, batch_name="batch2")  — run a second, independent batch
    run("meta")                                      — combine all stored exp1 batches
    run("meta", ds="cifar10")                        — combine only this dataset's batches
    run("figures")                                    — save PNGs from stored results
    run("summary")                                    — print paper-ready table

    Parameters
    ──────────
    seeds      : list of ints, default [42,7,123,0,99]
    datasets   : tuple of dataset names to run exp1 on, e.g. ("cifar10","svhn")
    batch_name : optional label so you can run exp1 twice and later call
                 run("meta") to combine the two independent batches properly
    ds         : restrict run("meta") to one dataset

    Example — replicate your two-batch design from before:
        run("exp1", seeds=[42,7,123,0,99,11,22,33,44,55], batch_name="batch1")
        run("exp1", seeds=[100,101,102,103,104,105,106,107,108,109], batch_name="batch2")
        run("meta")
    """
    if seeds is None:
        seeds = [42, 7, 123, 0, 99]
    if isinstance(datasets, str):
        datasets = (datasets,)

    if cmd == "smoke":
        exp_smoke(); return
    if cmd == "summary":
        exp_summary(); return
    if cmd == "figures":
        exp_figures(); return
    if cmd == "meta":
        exp_meta(ds=ds); return
    if cmd == "exp1":
        for d in datasets:
            exp1(seeds, synthetic=synthetic, ds=d, batch_name=batch_name)
        return
    raise ValueError(f"Unknown cmd '{cmd}'. Choose: smoke exp1 meta figures summary")


if __name__ == "__main__":
    if _in_notebook():
        print("Notebook mode. Call run('smoke'), run('exp1', seeds=[...]), etc.")
    else:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd", choices=["smoke", "exp1", "meta", "figures", "summary"])
        ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123, 0, 99])
        ap.add_argument("--datasets", type=str, nargs="+", default=["cifar10"])
        ap.add_argument("--batch_name", type=str, default=None)
        ap.add_argument("--ds", type=str, default=None)
        ap.add_argument("--synthetic", action="store_true")
        a = ap.parse_args()
        run(a.cmd, a.seeds, a.synthetic, tuple(a.datasets), a.batch_name, a.ds)
