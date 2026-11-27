#!/usr/bin/env python3
"""
    One experiment end to end: train a LoRA (src/train.py), answer MIHBench with it (src/infer_mihbench.py
    --adapter), then score and plot every run of the method (src/eval_mihbench.py --method). Each step is its own
    process, so training's GPU memory is free before inference, and each runs exactly as when called by hand; the
    experiment stops at the first step that fails.
      runs/<method>/<model>[-<tag>]/               LoRA weights, args.json, log.jsonl (src/train.py)
      results/mihbench/<method>/<model>[-<tag>]/   MIHBench answers and metrics.json
      results/mihbench/<method>/summary.csv        every run of the method x task x subset, and
      figures/mihbench/<method>/                   its plots
    Options not listed below go to src/train.py unchanged (--train, --epochs, --batch-size, --lambda-flip, ...).
      --mihbench-tasks   MIHBench tasks (default all five)
      --mihbench-limit   only the first N questions of each task (quick checks; a later run without it resumes)
      --vanilla          also answer with the untrained model (results/mihbench/vanilla/<model>/, same prompt and
                         precision) and score method vanilla: the baseline (inference resumes, so done once)
      --skip-train       the run folder exists already: only MIHBench
    InternVL answers MIHBench with sdpa attention (infer_mihbench.py --sdpa, folder suffix -sdpa), as it trains: the
    official eager attention builds a tokens x tokens matrix and takes ~4 s per question at ~10k tokens.
    python src/experiment.py --model qwen2.5-vl-3b --method cf-invariant --train pilot_train --tag pilot --vanilla
    python src/experiment.py --model qwen2.5-vl-3b --method sft --max-steps 2 --eval-groups 2 --tag smoke \\
        --mihbench-tasks existence_random --mihbench-limit 5                       # quick check of the chain
"""
import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
PROJECT = SRC.parent


def run(script, *args):
    """Run src/<script> with this Python from the project folder; stop the experiment if it fails."""
    cmd = [sys.executable, str(SRC / script), *map(str, args)]
    print(f"\n$ python src/{script} {' '.join(map(str, args))}", flush=True)
    code = subprocess.call(cmd, cwd=PROJECT)
    if code:
        sys.exit(f"src/{script} failed (exit {code}); experiment stopped")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--tag")
    ap.add_argument("--mihbench-tasks", nargs="+", default=["all"])
    ap.add_argument("--mihbench-limit", type=int)
    ap.add_argument("--vanilla", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args, train_args = ap.parse_known_args()
    if args.skip_train and train_args:
        ap.error(f"--skip-train with training options {train_args}")

    run_dir = Path("runs") / args.method / (args.model + (f"-{args.tag}" if args.tag else ""))  # as src/train.py
    if not args.skip_train:
        run("train.py", "--model", args.model, "--method", args.method, *(["--tag", args.tag] if args.tag else []),
            *train_args)
    mihbench = ["--task", *args.mihbench_tasks] + (["--limit", args.mihbench_limit] if args.mihbench_limit else [])
    if args.model.startswith("internvl"):
        mihbench.append("--sdpa")
    run("infer_mihbench.py", "--adapter", run_dir, *mihbench)
    run("eval_mihbench.py", "--method", args.method)
    if args.vanilla:
        run("infer_mihbench.py", "--model", args.model, *mihbench)
        run("eval_mihbench.py", "--method", "vanilla")
    print(f"\ndone: {run_dir}/, results/mihbench/{args.method}/{run_dir.name}{'-sdpa' if '--sdpa' in mihbench else ''}/, "
          f"results/mihbench/{args.method}/summary.csv")


if __name__ == "__main__":
    main()
