"""Ablation harness: does each mechanism actually help, or is it decoration?

Runs every variant across several seeds on the same panels and reports best out
of sample Sharpe, discovery speed, and calibration. Runs are sequential and in
process so memory stays bounded and nothing overlaps a heavy run.

  python ablate.py                         # clean synthetic data
  python ablate.py --data adversarial      # toxic data stress test
  python ablate.py --gens 80 --seeds 1 2 3 4 5 6
"""
import argparse
import contextlib
import io
import os
import shutil
import statistics
import tempfile

import numpy as np

import alpha_evolver as ae

np.seterr(all="ignore")  # extreme adversarial days overflow harmlessly; quiet it

ORDER = ["plain", "novelty", "surprise_mono", "surprise_noquar", "quar_mono",
         "with_lp", "ours"]


def one_run(variant, seed, data, gens, pop):
    d = tempfile.mkdtemp(prefix="ablate_")
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s = ae.run(generations=gens, pop=pop, seed=seed, data=data,
                       mem_path=os.path.join(d, "memory.json"),
                       history_path=os.path.join(d, "history.csv"),
                       curve_path=os.path.join(d, "curve.png"),
                       variant=variant, risk="max")
        h = s["history"]
        pes = [row[6] for row in h]            # mean_abs_pe per generation
        traj = [row[1] for row in h]           # best_oos_so_far per generation
        best = s["best_oos"]
        calib = (statistics.mean(pes[:5]) - statistics.mean(pes[-5:])
                 if len(pes) >= 10 else 0.0)
        target = 0.9 * best
        g90 = next((row[0] for row in h if row[1] >= target), len(h) - 1)
        cleangen = float("nan")
        overfit = float("nan")
        if data == "adversarial":
            # Re-test every trusted alpha on a CLEAN panel (no trap), independent
            # of the quarantine rule. cleangen: how well the memory's top alphas
            # by strength generalize (the downstream payoff). overfit: fraction of
            # trusted alphas that looked strong in sample but die on clean data
            # (the junk the firewall is supposed to keep out).
            clean = ae.synthetic_panel(seed=seed)
            csplit = int(0.70 * clean["returns"].shape[0])
            trusted = [r for r in s["memory"].items.values()
                       if r.get("status", "trusted") == "trusted"]
            clean_oos = {id(r): ae.backtest(r["expr"], clean, csplit)["oos_sharpe"]
                         for r in trusted}
            top = sorted(trusted, key=lambda r: -r.get("strength", 0.0))[:8]
            if top:
                cleangen = statistics.mean(clean_oos[id(r)] for r in top)
            scored = [r for r in trusted if r["stats"]["is_sharpe"] >= 1.0]
            if scored:
                overfit = sum(1 for r in scored if clean_oos[id(r)] < 0.3) / len(scored)
        return {"best": best, "g90": g90, "calib": calib, "median": h[-1][3],
                "hof": len(s["hof"]), "cleangen": cleangen, "overfit": overfit}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="synthetic")
    p.add_argument("--gens", type=int, default=60)
    p.add_argument("--pop", type=int, default=60)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--variants", nargs="+", default=ORDER)
    args = p.parse_args()

    print(f"ablation  data {args.data}  gens {args.gens}  pop {args.pop}  "
          f"seeds {args.seeds}", flush=True)
    results = {v: [] for v in args.variants}
    for v in args.variants:
        for seed in args.seeds:
            r = one_run(v, seed, args.data, args.gens, args.pop)
            results[v].append(r)
            cg = ("" if r["cleangen"] != r["cleangen"] else
                  f"  cleanGen {r['cleangen']:+.2f}  overfit {r['overfit']:.2f}")
            print(f"  {v:16s} seed {seed}: best {r['best']:+.3f}  "
                  f"g90 {r['g90']:3d}  calib {r['calib']:+.3f}{cg}", flush=True)

    print(f"\n==== ABLATION SUMMARY  data={args.data}  "
          f"(mean over {len(args.seeds)} seeds) ====", flush=True)
    print(f"{'variant':16s} {'best_oos mean±sd':>18s} {'g90':>6s} "
          f"{'calib':>7s} {'cleanGen':>9s} {'overfit%':>9s}", flush=True)
    for v in args.variants:
        rs = results[v]
        bs = [r["best"] for r in rs]
        sd = statistics.pstdev(bs) if len(bs) > 1 else 0.0
        cg_vals = [r["cleangen"] for r in rs if r["cleangen"] == r["cleangen"]]
        of_vals = [r["overfit"] for r in rs if r["overfit"] == r["overfit"]]
        cg = f"{statistics.mean(cg_vals):+8.2f}" if cg_vals else "      --"
        of = f"{statistics.mean(of_vals) * 100:8.0f}" if of_vals else "      --"
        print(f"{v:16s} {statistics.mean(bs):+7.3f} ± {sd:5.3f}  "
              f"{statistics.mean(r['g90'] for r in rs):6.1f} "
              f"{statistics.mean(r['calib'] for r in rs):+7.3f} "
              f"{cg} {of}", flush=True)


if __name__ == "__main__":
    main()
