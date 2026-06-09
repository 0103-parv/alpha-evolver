"""Ablations for the Max Cut creativity laboratory."""

import argparse
import json
import statistics
import time

import maxcut_lab as lab


SURPRISE_VARIANTS = (
    "plain",
    "surprise_noquar",
    "quarantine_only",
    "quar_mono",
    "inverted_u",
)
STOP_POLICIES = ("fixed", "rules", "learned")


def print_reference(seeds):
    rows = []
    for seed in seeds:
        domain = lab.MaxCutDomain(seed)
        rows.append(domain.evaluate_sealed(lab.baseline_program())["mean_quality"])
    print(
        f"greedy reference sealed {statistics.mean(rows):.4f} "
        f"+/- {statistics.pstdev(rows):.4f}",
        flush=True,
    )


def one_run(seed, generations, pop, variant, stop_policy):
    start = time.time()
    result = lab.run(
        generations=generations,
        pop=pop,
        seed=seed,
        max_minutes=90,
        variant=variant,
        stop_policy=stop_policy,
        memory_path=None,
        history_path=None,
        replay=False,
        quiet=True,
        memory_obj=lab.GraphMemory(),
        persist=False,
    )
    history = result["history"]
    pe = [row["mean_abs_pe"] for row in history]
    calibration = (
        statistics.mean(pe[:5]) - statistics.mean(pe[-5:])
        if len(pe) >= 10 else 0.0
    )
    return {
        "seed": seed,
        "variant": variant,
        "stop_policy": stop_policy,
        "train": result["best"]["train"]["mean_quality"],
        "dev": result["best"]["dev"]["mean_quality"],
        "sealed": result["sealed"]["mean_quality"],
        "family_std": result["sealed"]["family_std"],
        "generations": len(history),
        "calibration": calibration,
        "seconds": time.time() - start,
        "stop_reason": result["stop_reason"],
    }


def _summary(rows, key):
    values = [row[key] for row in rows]
    return statistics.mean(values), statistics.pstdev(values)


def run_group(labels, seeds, generations, pop, kind):
    rows = []
    for label in labels:
        for seed in seeds:
            variant = label if kind == "surprise" else "ours"
            stop_policy = "fixed" if kind == "surprise" else label
            row = one_run(seed, generations, pop, variant, stop_policy)
            rows.append(row)
            print(
                f"  {label:18s} seed {seed}: sealed {row['sealed']:.4f}  "
                f"dev {row['dev']:.4f}  gens {row['generations']:3d}  "
                f"calib {row['calibration']:+.4f}  {row['seconds']:.1f}s",
                flush=True,
            )
    print(f"\n==== MAX CUT {kind.upper()} SUMMARY ====", flush=True)
    for label in labels:
        selected = [
            row for row in rows
            if (row["variant"] if kind == "surprise" else row["stop_policy"]) == label
        ]
        sealed, sealed_sd = _summary(selected, "sealed")
        gens, gens_sd = _summary(selected, "generations")
        calibration, _ = _summary(selected, "calibration")
        print(
            f"{label:18s} sealed {sealed:.4f} +/- {sealed_sd:.4f}  "
            f"gens {gens:.1f} +/- {gens_sd:.1f}  calib {calibration:+.4f}",
            flush=True,
        )
    if kind == "surprise":
        inverted = {row["seed"]: row for row in rows if row["variant"] == "inverted_u"}
        quarantine = {
            row["seed"]: row for row in rows if row["variant"] == "quarantine_only"}
        differences = [
            inverted[seed]["sealed"] - quarantine[seed]["sealed"]
            for seed in seeds
        ]
        print(
            f"inverted U minus quarantine only: mean {statistics.mean(differences):+.4f}, "
            f"wins {sum(value > 0 for value in differences)}/{len(differences)}",
            flush=True,
        )
    return rows


def run_memory_experiment(seeds, generations, pop):
    """Build corroborated memory, then measure matched cold versus warm lift."""
    memory = lab.GraphMemory()
    for seed in seeds:
        result = lab.run(
            generations=generations,
            pop=pop,
            seed=seed,
            max_minutes=90,
            variant="ours",
            stop_policy="fixed",
            memory_path=None,
            history_path=None,
            replay=False,
            quiet=True,
            memory_obj=memory,
            persist=False,
        )
        memory = result["memory"]
        memory.distill_cards()
        trusted = sum(card["status"] == "trusted" for card in memory.cards)
        print(
            f"  memory build seed {seed}: episodes {len(memory.episodes)}  "
            f"trusted cards {trusted}",
            flush=True,
        )
    rows = []
    for seed in seeds:
        result = lab.paired_memory_replay(memory, seed + 50000)
        rows.append(result)
        print(
            f"  replay seed {seed}: lift {result['lift']:+.5f}  "
            f"diagnosis {result['diagnosis']}  "
            f"cards {result['warm_injection']['retrieved_cards']}",
            flush=True,
        )
    lifts = [row["lift"] for row in rows]
    print(
        f"\n==== MAX CUT MEMORY SUMMARY ====\n"
        f"mean lift {statistics.mean(lifts):+.5f}  "
        f"positive {sum(value > 0 for value in lifts)}/{len(lifts)}",
        flush=True,
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment",
                        choices=("surprise", "stopping", "memory", "both"),
                        default="both")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    all_rows = []
    print(
        f"graph ablation  experiment {args.experiment}  gens {args.generations}  "
        f"pop {args.pop}  seeds {args.seeds}",
        flush=True,
    )
    print_reference(args.seeds)
    if args.experiment in ("surprise", "both"):
        all_rows.extend(run_group(
            SURPRISE_VARIANTS, args.seeds, args.generations, args.pop, "surprise"))
    if args.experiment in ("stopping", "both"):
        all_rows.extend(run_group(
            STOP_POLICIES, args.seeds, args.generations, args.pop, "stopping"))
    if args.experiment == "memory":
        all_rows.extend(run_memory_experiment(
            args.seeds, args.generations, args.pop))
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(all_rows, handle, indent=2)


if __name__ == "__main__":
    main()
