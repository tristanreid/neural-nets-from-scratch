"""
Experiment runner for the MoE blog post (Part 4).

Runs each experiment × seed as an isolated subprocess under a named run directory.
Full output goes to per-experiment log files; console shows only milestones.
Three seeds (3407, 42, 7) are run per experiment for statistical robustness.

Usage:
    python run_experiments.py --run part4_v2        # run all under runs/part4_v2/
    python run_experiments.py --run part4_v2 --force  # re-run even if complete
    python run_experiments.py --list                # show status of all runs
    python run_experiments.py --only moe_top1_aux   # run one experiment (all seeds)
    python run_experiments.py --skip-training       # analysis only (models must exist)
    python run_experiments.py --verbose             # show all output on console
    python run_experiments.py --max-steps 50        # override steps (for testing)

Directory structure:
    runs/
      part4_v2/                     # one directory per run codename
        config.json                 # snapshot of experiment configs
        summary.json                # aggregated results (mean ± std across seeds)
        dense_baseline/
          seed_3407/
            train.log               # full training output
            samples.log             # generated samples
            domains.log             # per-domain analysis
            domain_analysis.json    # machine-readable per-domain metrics
            model.pt                # best checkpoint
          seed_42/
            ...
          seed_7/
            ...
        moe_top1_aux/
          seed_3407/
            train.log
            routing.log             # per-domain routing analysis
            routing_analysis.json   # machine-readable routing data
            domains.log
            samples.log
            model.pt
          ...
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

SEEDS = [3407, 42, 7]  # three seeds for statistical robustness

COMMON = dict(
    input_file="data/domain_mix_train.txt",
    n_layer=2,
    n_head=4,
    n_embd=48,
    device="cpu",
    num_workers=0,
    batch_size=32,
    learning_rate=5e-4,
    max_steps=20000,
)

EXPERIMENTS = [
    {
        "name": "dense_baseline",
        "description": "Dense transformer (standard FFN, no MoE)",
        "type": "transformer",
    },
    {
        "name": "moe_top1_aux",
        "description": "MoE — 4 experts, top-1 routing, with load balancing",
        "type": "moe",
        "num_experts": 4,
        "top_k": 1,
        "aux_loss_coef": 0.01,
    },
    {
        "name": "moe_top1_noaux",
        "description": "MoE — 4 experts, top-1 routing, NO load balancing (collapse demo)",
        "type": "moe",
        "num_experts": 4,
        "top_k": 1,
        "aux_loss_coef": 0.0,
    },
    {
        "name": "moe_top2_aux",
        "description": "MoE — 4 experts, top-2 routing, with load balancing",
        "type": "moe",
        "num_experts": 4,
        "top_k": 2,
        "aux_loss_coef": 0.01,
    },
]

# Post-training analysis steps
ANALYSIS = [
    {
        "name": "routing",
        "description": "Routing analysis per domain",
        "applies_to": ["moe_top1_aux", "moe_top1_noaux", "moe_top2_aux"],
        "extra_args": ["--analyze-routing", "--resume"],
    },
    {
        "name": "domains",
        "description": "Per-domain loss + arithmetic accuracy",
        "applies_to": ["dense_baseline", "moe_top1_aux", "moe_top1_noaux", "moe_top2_aux"],
        "extra_args": ["--analyze-domains", "--resume"],
    },
    {
        "name": "samples",
        "description": "Sample generation",
        "applies_to": ["dense_baseline", "moe_top1_aux", "moe_top1_noaux", "moe_top2_aux"],
        "extra_args": ["--sample-only", "--resume"],
    },
]

# Lines matching these patterns are shown on the console during training.
# Everything else is log-only (unless --verbose).
CONSOLE_PATTERNS = [
    re.compile(r"^(number of|MoE transformer|total model|dataset determined|split up)"),
    re.compile(r"^step \d+ train loss"),     # eval checkpoints
    re.compile(r"^test loss"),               # new best
    re.compile(r"^\s+layer \d+ routing"),    # expert utilization
    re.compile(r"^-{10,}"),                  # sample separators
    re.compile(r"^\d+ samples that are"),    # sample categories
    re.compile(r"^Training complete"),
    re.compile(r"^loading model"),
    re.compile(r"^Analyzing"),
    re.compile(r"^\s+(name|arith|code):"),   # routing analysis domain headers
    re.compile(r"^\s+layer_\d+:"),           # routing analysis per-layer
    re.compile(r"^\s+Routing analysis saved"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_cmd(exp, out_dir, seed=None, extra_args=None):
    """Build the python command for a training or analysis run."""
    cfg = {**COMMON, **exp}
    cmd = [sys.executable, "-u", "makemore_moe.py"]
    cmd += ["-i", cfg["input_file"]]
    cmd += ["--type", cfg["type"]]
    cmd += ["--n-layer", str(cfg["n_layer"])]
    cmd += ["--n-head", str(cfg["n_head"])]
    cmd += ["--n-embd", str(cfg["n_embd"])]
    cmd += ["--device", cfg["device"]]
    cmd += ["--num-workers", str(cfg["num_workers"])]
    cmd += ["-b", str(cfg["batch_size"])]
    cmd += ["-l", str(cfg["learning_rate"])]
    cmd += ["--max-steps", str(cfg["max_steps"])]
    cmd += ["-o", out_dir]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if cfg["type"] == "moe":
        cmd += ["--num-experts", str(cfg["num_experts"])]
        cmd += ["--top-k", str(cfg["top_k"])]
        cmd += ["--aux-loss-coef", str(cfg["aux_loss_coef"])]
    if extra_args:
        cmd += extra_args
    return cmd


def exp_dir(run_dir, exp_name, seed=None):
    """Get the output directory for an experiment, optionally per-seed."""
    if seed is not None:
        return os.path.join(run_dir, exp_name, f"seed_{seed}")
    return os.path.join(run_dir, exp_name)


def model_exists_for_seed(exp_name, run_dir, seed):
    """Check if an experiment+seed has a saved model."""
    return os.path.exists(os.path.join(exp_dir(run_dir, exp_name, seed), "model.pt"))


def should_show(line):
    """Return True if this line should appear on the console (non-verbose mode)."""
    return any(p.search(line) for p in CONSOLE_PATTERNS)


def run_cmd(cmd, log_path, verbose=False):
    """
    Run a command, saving full output to log_path.
    Console shows only milestone lines (or everything if verbose).
    Returns (exit_code, key_lines) where key_lines are the filtered lines.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    key_lines = []

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, universal_newlines=True,
        )
        for line in proc.stdout:
            log_f.write(line)
            if verbose or should_show(line):
                sys.stdout.write(line)
                key_lines.append(line.rstrip())
        proc.wait()

    return proc.returncode, key_lines


def extract_metrics(log_path):
    """Parse a training log for final train/test loss and timing."""
    metrics = {}
    try:
        with open(log_path) as f:
            for line in f:
                # Match eval lines: "step 4500 train loss: 1.2345 test loss: 1.3456"
                m = re.match(r"step (\d+) train loss: ([\d.]+) test loss: ([\d.]+)", line)
                if m:
                    metrics["last_eval_step"] = int(m.group(1))
                    metrics["train_loss"] = float(m.group(2))
                    metrics["test_loss"] = float(m.group(3))
                # Match best loss: "test loss X.XXXX is the best so far"
                m = re.match(r"test loss ([\d.]+) is the best", line)
                if m:
                    metrics["best_test_loss"] = float(m.group(1))
                # Match param count
                m = re.search(r"total model parameters: ([\d,]+)", line)
                if m:
                    metrics["total_params"] = int(m.group(1).replace(",", ""))
    except FileNotFoundError:
        pass
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MoE experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --run part4_v1
  python run_experiments.py --run part4_v1 --only moe_top1_aux
  python run_experiments.py --run part4_v1 --skip-training
  python run_experiments.py --list
        """,
    )
    parser.add_argument("--run", type=str, default=None,
                        help="Run codename (default: timestamp-based)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run experiments even if model.pt exists")
    parser.add_argument("--list", action="store_true",
                        help="List all runs and their status")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only the named experiment")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, only run analysis")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max_steps for all experiments (for testing)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show all output on console (not just milestones)")
    args = parser.parse_args()

    # --- Override max_steps if requested ---
    if args.max_steps is not None:
        COMMON["max_steps"] = args.max_steps

    # --- List mode: show all existing runs ---
    if args.list:
        runs_dir = "runs"
        if not os.path.isdir(runs_dir):
            print("No runs yet. Use --run <codename> to start one.")
            return
        print("Runs:")
        for entry in sorted(os.listdir(runs_dir)):
            run_path = os.path.join(runs_dir, entry)
            if not os.path.isdir(run_path):
                continue
            summary_path = os.path.join(run_path, "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    summary = json.load(f)
                print(f"\n  {entry}/")
                for name, info in summary.items():
                    if "test_loss_mean" in info:
                        print(f"    {name}: {info['test_loss_mean']:.4f} ± {info.get('test_loss_std', 0):.4f} ({info.get('seeds', '?')} seeds, {info.get('total_time', 0):.0f}s)")
                    elif "test_loss" in info:
                        print(f"    {name}: test_loss={info['test_loss']:.4f}")
            else:
                print(f"\n  {entry}/ (no summary)")
        return

    # --- Determine run directory ---
    run_name = args.run or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Run: {run_name}")
    print(f"Dir: {run_dir}/")
    print()

    # Save config snapshot
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump({
                "common": COMMON,
                "experiments": EXPERIMENTS,
                "analysis": ANALYSIS,
                "created": datetime.now().isoformat(),
            }, f, indent=2)

    # Filter experiments if --only is specified
    experiments = EXPERIMENTS
    if args.only:
        experiments = [e for e in experiments if e["name"] == args.only]
        if not experiments:
            print(f"ERROR: No experiment named '{args.only}'")
            print(f"Available: {', '.join(e['name'] for e in EXPERIMENTS)}")
            sys.exit(1)

    seeds = SEEDS
    summary = {}
    total_runs = len(experiments) * len(seeds)

    # --- Training phase ---
    if not args.skip_training:
        print("=" * 60)
        print(f"TRAINING ({len(experiments)} experiments × {len(seeds)} seeds = {total_runs} runs)")
        print("=" * 60)

        run_num = 0
        for exp in experiments:
            name = exp["name"]
            for seed in seeds:
                run_num += 1
                out = exp_dir(run_dir, name, seed)
                log_path = os.path.join(out, "train.log")

                print(f"\n[{run_num}/{total_runs}] {name} (seed={seed})")
                print(f"  {exp['description']}")

                if model_exists_for_seed(name, run_dir, seed) and not args.force:
                    print(f"  SKIP (model exists)")
                    metrics = extract_metrics(log_path)
                    summary.setdefault(name, {})[seed] = {"status": "skipped", "time": 0, **metrics}
                    continue

                cmd = build_cmd(exp, out, seed=seed)
                print(f"  Log: {log_path}")

                t0 = time.time()
                rc, _ = run_cmd(cmd, log_path, verbose=args.verbose)
                elapsed = time.time() - t0

                metrics = extract_metrics(log_path)

                if rc != 0:
                    print(f"\n  FAILED (exit code {rc}) after {elapsed:.0f}s")
                    print(f"  Full output: {log_path}")
                    summary.setdefault(name, {})[seed] = {"status": "failed", "time": elapsed, "exit_code": rc}
                else:
                    loss_str = f" — test_loss={metrics.get('test_loss', '?')}" if metrics else ""
                    print(f"\n  DONE in {elapsed:.0f}s{loss_str}")
                    summary.setdefault(name, {})[seed] = {"status": "done", "time": elapsed, **metrics}

    # --- Analysis phase ---
    print()
    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    for analysis in ANALYSIS:
        for exp in experiments:
            name = exp["name"]
            if name not in analysis["applies_to"]:
                continue
            # Run analysis on first seed only (routing/domain analysis)
            seed = seeds[0]
            out = exp_dir(run_dir, name, seed)
            if not os.path.exists(os.path.join(out, "model.pt")):
                print(f"\n  Skip {analysis['name']} for {name} — no model")
                continue

            print(f"\n  {analysis['name']}: {name} (seed={seed})")
            cmd = build_cmd(exp, out, seed=seed, extra_args=analysis["extra_args"])
            log_path = os.path.join(out, f"{analysis['name']}.log")

            t0 = time.time()
            rc, _ = run_cmd(cmd, log_path, verbose=args.verbose)
            elapsed = time.time() - t0

            status = "done" if rc == 0 else f"failed (exit {rc})"
            print(f"  {status} ({elapsed:.0f}s) — {log_path}")

    # --- Aggregate summary ---
    agg_summary = {}
    for name, seed_results in summary.items():
        test_losses = [v["test_loss"] for v in seed_results.values()
                       if "test_loss" in v]
        train_losses = [v["train_loss"] for v in seed_results.values()
                        if "train_loss" in v]
        total_time = sum(v.get("time", 0) for v in seed_results.values())

        agg = {"seeds": len(seed_results), "total_time": round(total_time, 1)}
        if test_losses:
            import statistics
            agg["test_loss_mean"] = round(statistics.mean(test_losses), 4)
            agg["test_loss_std"] = round(statistics.stdev(test_losses), 4) if len(test_losses) > 1 else 0.0
            agg["test_losses"] = [round(l, 4) for l in test_losses]
        if train_losses:
            agg["train_loss_mean"] = round(statistics.mean(train_losses), 4)
            agg["train_loss_std"] = round(statistics.stdev(train_losses), 4) if len(train_losses) > 1 else 0.0
        if seed_results:
            first = list(seed_results.values())[0]
            if "total_params" in first:
                agg["total_params"] = first["total_params"]
        agg["per_seed"] = {str(k): v for k, v in seed_results.items()}
        agg_summary[name] = agg

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(agg_summary, f, indent=2)

    print()
    print("=" * 60)
    print(f"COMPLETE — {run_name}")
    print("=" * 60)
    print()

    # Show aggregated results
    print(f"  {'Experiment':<22s} {'Params':>8s} {'Test Loss':>18s} {'Time':>8s}")
    print(f"  {'-'*22} {'-'*8} {'-'*18} {'-'*8}")
    for name, agg in agg_summary.items():
        params = f"{agg.get('total_params', 0):,}"
        if "test_loss_mean" in agg:
            loss_str = f"{agg['test_loss_mean']:.4f} ± {agg['test_loss_std']:.4f}"
        else:
            loss_str = "—"
        time_str = f"{agg['total_time']:.0f}s"
        print(f"  {name:<22s} {params:>8s} {loss_str:>18s} {time_str:>8s}")

    print(f"\n  Directory: {run_dir}/")
    print(f"  Summary:   {summary_path}")
    print(f"\n  To re-run analysis only:")
    print(f"    python run_experiments.py --run {run_name} --skip-training")


if __name__ == "__main__":
    main()
