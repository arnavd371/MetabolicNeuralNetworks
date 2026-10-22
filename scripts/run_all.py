"""Driver for the reproduction. Usage:  python run_all.py <smoke|part1|part2|part3|svhn> [--out DIR]

part1: the faithful protocol, calling the unmodified metnn_exp1.run(...)
part2: instrumented runs of the original curriculum (20 seeds) + controls
part3: binary per-pass activation rate, original curriculum, 10 seeds
svhn:  unchanged exp1 on SVHN (20 seeds) + instrumented SVHN runs with per-check logs
CSV files are rewritten after every seed so partial results survive a timeout.
"""
import argparse, csv, io, json, os, platform, subprocess, sys, time

B1 = [42, 7, 123, 0, 99, 11, 22, 33, 44, 55]
B2 = list(range(100, 110))


class Tee(io.TextIOBase):
    def __init__(self, *s): self.s = s
    def write(self, x):
        for s in self.s: s.write(x); s.flush()
        return len(x)
    def flush(self):
        for s in self.s: s.flush()


def env_info():
    import torch, torchvision, numpy, scipy
    info = dict(python=platform.python_version(), torch=torch.__version__,
                torchvision=torchvision.__version__, numpy=numpy.__version__,
                scipy=scipy.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception as e:
        info["nvidia_smi"] = repr(e)
    return info


def write_csv(path, rows):
    if not rows: return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def part1(out):
    import numpy as np
    import metnn_exp1 as M
    M.run("smoke")
    t = time.time()
    M.run("exp1", seeds=B1, batch_name="batch1")
    M.run("exp1", seeds=B2, batch_name="batch2")
    M.run("meta"); M.run("figures"); M.run("summary")
    print(f"[part1 wall time {time.time()-t:.0f}s]")
    rows = []
    for key in ("exp1_cifar10_batch1", "exp1_cifar10_batch2"):
        r = M.RESULTS[key]
        for s, e, h, a in zip(r["seeds"], r["easy"], r["hard"], r["accs"]):
            rows.append(dict(batch=r["batch_name"], seed=s, easy_to_med=e,
                             med_to_hard=h, acc=a))
    write_csv(f"{out}/part1_per_seed.csv", rows)
    json.dump(M.RESULTS, open(f"{out}/part1_RESULTS.json", "w"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def run_set(out, tag, jobs):
    """jobs: list of (curriculum, rate_mode, seed)."""
    import metnn_instr as I
    seeds_rows, check_rows, epoch_rows = [], [], []
    for cur, rm, s in jobs:
        r = I.run_curriculum_logged(s, curriculum=cur, rate_mode=rm)
        row = dict(curriculum=cur, rate_mode=rm, seed=s, acc=r["acc"],
                   total_steps=r["total_steps"], seconds=round(r["seconds"], 1))
        for k, v in r["td"].items(): row[f"td_{k}"] = v
        for k, v in r["alive"].items(): row[f"alive_end_{k}"] = v
        seeds_rows.append(row); check_rows += r["checks"]; epoch_rows += r["epochs"]
        print(f"  [{tag}] {cur:16s} {rm:6s} seed {s:3d}  td={r['td']}  "
              f"acc {r['acc']:.1f}%  ({r['seconds']:.0f}s)", flush=True)
        write_csv(f"{out}/{tag}_per_seed.csv", seeds_rows)
        write_csv(f"{out}/{tag}_per_check.csv", check_rows)
        write_csv(f"{out}/{tag}_per_epoch.csv", epoch_rows)


def svhn(out, seeds):
    """B1: original exp1 on SVHN (unchanged code), then instrumented SVHN runs."""
    import math
    import torch
    import metnn_exp1 as M, metnn_instr as I
    (xtr, ytr), _ = M.get_dataset("svhn")
    counts = torch.bincount(ytr.cpu(), minlength=10).tolist()
    print("SVHN train class counts (digit: n):", dict(enumerate(counts)))
    step, sched = 0, []
    for name, classes, epochs in M.CURRICULUM:
        n = sum(counts[c] for c in classes); spe = math.ceil(n / 128)
        e0 = (step + 1, step + spe)
        checks = [s for s in range(e0[0], e0[1] + 1) if s % 150 == 0]
        sched.append(dict(phase=name, digits=" ".join(map(str, classes)), n_samples=n,
                          steps_per_epoch=spe, epochs=epochs, phase_start=step + 1,
                          phase_end=step + spe * epochs, epoch0_start=e0[0], epoch0_end=e0[1],
                          epoch0_checks=" ".join(map(str, checks))))
        print(f"  {name:6s} digits {classes}  n={n}  steps/epoch={spe}  epoch0 steps "
              f"{e0[0]}-{e0[1]}  checks in epoch0: {checks}")
        step += spe * epochs
    write_csv(f"{out}/svhn_schedule.csv", sched)
    M.run("exp1", seeds=seeds, datasets=("svhn",), batch_name="svhn_b1b2")
    r = M.RESULTS["exp1_svhn_svhn_b1b2"]
    write_csv(f"{out}/svhn_original_per_seed.csv",
              [dict(seed=s, easy_to_med=e, med_to_hard=h, acc=acc)
               for s, e, h, acc in zip(r["seeds"], r["easy"], r["hard"], r["accs"])])
    seeds_rows, check_rows, epoch_rows = [], [], []
    for s in seeds:
        x = I.run_curriculum_logged(s, ds="svhn")
        row = dict(seed=s, acc=x["acc"], total_steps=x["total_steps"])
        for k, v in x["td"].items(): row[f"td_{k}"] = v
        seeds_rows.append(row); check_rows += x["checks"]; epoch_rows += x["epochs"]
        print(f"  [svhn instr] seed {s:3d} td={x['td']} acc {x['acc']:.1f}%", flush=True)
        write_csv(f"{out}/svhn_per_seed.csv", seeds_rows)
        write_csv(f"{out}/svhn_per_check.csv", check_rows)
        write_csv(f"{out}/svhn_per_epoch.csv", epoch_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["smoke", "part1", "part2", "part3", "svhn"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--curricula", nargs="+",
                    default=["original", "noshift_matched", "equal_len", "reversed"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = open(f"{a.out}/{a.what}.txt", "a")
    sys.stdout = Tee(sys.__stdout__, log)
    info = env_info(); print(json.dumps(info, indent=1))
    json.dump(info, open(f"{a.out}/{a.what}_env.json", "w"), indent=1)

    if a.what == "smoke":
        import metnn_exp1 as M, metnn_instr as I
        M.run("smoke")
        if a.synthetic:      # local plumbing test only
            M.CURRICULUM[:] = [(n, c, 1) for n, c, _ in M.CURRICULUM]
            for k in I.CURRICULA:
                I.CURRICULA[k] = [(n, c, 1, nn) for n, c, _, nn in I.CURRICULA[k]]
        # timing + determinism: original vs instrumented, same seed
        t = time.time(); r0 = M._run_curriculum(42); t0 = time.time() - t
        r1 = I.run_curriculum_logged(42)
        print(f"original td={r0['td']} acc={r0['acc']:.2f} ({t0:.0f}s)")
        print(f"instr    td={r1['td']} acc={r1['acc']:.2f} ({r1['seconds']:.0f}s)")
        print("MATCH" if r0["td"] == r1["td"] and r0["acc"] == r1["acc"] else "MISMATCH")
        for cur in ("noshift_matched", "equal_len", "reversed"):
            r = I.run_curriculum_logged(42, curriculum=cur)
            print(f"{cur}: td={r['td']} steps={r['total_steps']} ({r['seconds']:.0f}s)")
        r = I.run_curriculum_logged(42, rate_mode="binary")
        print(f"binary: td={r['td']} ({r['seconds']:.0f}s)")
        write_csv(f"{a.out}/smoke_checks.csv", r1["checks"])
    elif a.what == "part1":
        part1(a.out)
    elif a.what == "part2":
        seeds = a.seeds or B1 + B2
        run_set(a.out, "part2", [(c, "frac", s) for c in a.curricula for s in seeds])
    elif a.what == "svhn":
        svhn(a.out, a.seeds or B1 + B2)
    elif a.what == "part3":
        seeds = a.seeds or B1
        run_set(a.out, "part3", [("original", "binary", s) for s in seeds])
    print("DONE", a.what)


if __name__ == "__main__":
    main()
