#!/usr/bin/env python3
"""
    Inference of 00_baseline_analyze (README.md): MIHBench existence under conditions A, B and C, one line per
    model call in results/00_baseline_analyze/<model>/raw/<cond>/<task>.jsonl (output.results in config.yaml).
      A  each image alone              "Is there a {obj} in this image?"            3 calls per question
      B  all 3 images, one asked       "Is there a {obj} in the {ordinal} image?"   3 calls per question
      C  all 3 images, the benchmark   the MIHBench question + the family's instruction (configs/prompts.yaml)
    A line: task, question_id, image_index (A, B), images, obj, prompt, response, first_token, logits.
    Re-running skips the calls already saved. It stops before loading the model if a saved call was asked with
    another prompt or images, or if raw/run.json records other model or decoding settings, so results made under
    different conditions never mix. Ground truth is not written here: analyze.py adds it.
    python experiments/00_baseline_analyze/infer.py --model qwen2.5-vl-3b                       # A, B, C, all tasks
    python experiments/00_baseline_analyze/infer.py --model qwen2.5-vl-3b --cond C --limit 20   # quick check
"""
import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from common.mihbench import load_questions, question_object  # noqa: E402
from common.models import load_model  # noqa: E402
from common.prompts import PROMPTS, fill  # noqa: E402
from common.runinfo import environment, model_commit, pick_device  # noqa: E402

CONDITIONS = ("A", "B", "C")


def load_config(path=HERE / "config.yaml"):
    return yaml.safe_load(path.read_text())


def build_calls(cond, task, questions, cfg, family):
    """The model calls of one condition on one task, in question order, with the input fields of their lines."""
    spec = cfg["conditions"][cond]
    calls = []
    for q in questions:
        obj = question_object(q["text"])
        if cond == "C":
            calls.append({"task": task, "question_id": q["question_id"], "images": q["image_list"], "obj": obj,
                          "prompt": fill(PROMPTS[family], q["text"])})
            continue
        if cond == "B" and len(q["image_list"]) > len(spec["ordinals"]):
            raise SystemExit(f"{task} {q['question_id']}: {len(q['image_list'])} images, "
                             f"but conditions.B.ordinals has {len(spec['ordinals'])}")
        for i, image in enumerate(q["image_list"]):
            prompt = spec["prompt"].replace("{obj}", obj)
            if cond == "B":
                prompt = prompt.replace("{ordinal}", spec["ordinals"][i])
            calls.append({"task": task, "question_id": q["question_id"], "image_index": i,
                          "images": [image] if cond == "A" else q["image_list"], "obj": obj, "prompt": prompt})
    return calls


def call_key(row):
    return row["question_id"], row.get("image_index")


def read_rows(path):
    """Lines saved so far, by call. Drops a line cut off by an interrupted run."""
    if not path.exists():
        return {}
    rows, lines = {}, path.read_text().splitlines()
    for line in lines:
        try:
            row = json.loads(line)
            rows[call_key(row)] = row
        except (json.JSONDecodeError, KeyError):
            pass
    if len(rows) != len(lines):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
    return rows


def check_saved(path, calls, saved):
    """Stop if a saved call had other inputs than the same call now: resuming would mix two conditions."""
    for call in calls:
        row = saved.get(call_key(call))
        if row and (row["prompt"], row["images"]) != (call["prompt"], call["images"]):
            raise SystemExit(f"{path}: question {call['question_id']} was asked {row['prompt']!r} on {row['images']}, "
                             f"config.yaml now gives {call['prompt']!r} on {call['images']}. "
                             "Move the results folder away or restore the config.")


def read_run(path, settings):
    """raw/run.json: the settings every saved line was made with, and one entry per invocation."""
    run = json.loads(path.read_text()) if path.exists() else {"settings": settings, "runs": []}
    if run["settings"] != settings:
        raise SystemExit(f"{path}: saved results were made with\n  {run['settings']}\nnot\n  {settings}\n"
                         "Move the results folder away or restore the config.")
    return run


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(cfg["models"]))
    ap.add_argument("--cond", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    ap.add_argument("--task", nargs="+", choices=cfg["tasks"], default=cfg["tasks"])
    ap.add_argument("--limit", type=int, help="only the first N questions of each task (quick check)")
    args = ap.parse_args()
    conds, tasks = list(dict.fromkeys(args.cond)), list(dict.fromkeys(args.task))

    model_cfg, decoding = cfg["models"][args.model], cfg["decoding"]
    if decoding["do_sample"] is not False:
        raise SystemExit("config.yaml decoding.do_sample: only greedy decoding (false) is supported")
    if not all(isinstance(t, str) for t in decoding["logit_tokens"]):  # YAML reads bare Yes/No as booleans
        raise SystemExit(f"config.yaml decoding.logit_tokens: {decoding['logit_tokens']}, quote them: \"Yes\"")
    model_dir = ROOT / model_cfg["path"]
    if not model_dir.is_dir():
        raise SystemExit(f"{model_dir} not found: python scripts/download_models.py --model {args.model}")
    out_dir = ROOT / cfg["output"]["results"].format(model=args.model) / "raw"
    image_dir = ROOT / cfg["data"]["images"]

    # Everything that can fail without the model is checked first.
    plan = []
    for task in tasks:
        questions = load_questions(ROOT / cfg["data"]["questions"].format(task=task))[:args.limit]
        missing = sorted({im for q in questions for im in q["image_list"] if not (image_dir / im).is_file()})
        if missing:
            raise SystemExit(f"{task}: {len(missing)} images missing in {image_dir}, e.g. {missing[0]}")
        for cond in conds:
            path = out_dir / cond / f"{task}.jsonl"
            calls = build_calls(cond, task, questions, cfg, model_cfg["family"])
            saved = read_rows(path)
            check_saved(path, calls, saved)
            plan.append((cond, task, path, [c for c in calls if call_key(c) not in saved], len(calls)))
    settings = {"model": args.model, **model_cfg, "model_commit": model_commit(model_dir), "decoding": decoding}
    run_path = out_dir / "run.json"
    run = read_run(run_path, settings)
    if not any(todo for _, _, _, todo, _ in plan):
        print(f"Nothing to do: every call is already in {out_dir}")
        return

    device = pick_device()
    model = load_model({**model_cfg, "path": str(model_dir)}, decoding["logit_tokens"], device)
    run["runs"].append({"conditions": conds, "tasks": tasks, "limit": args.limit, **environment(device)})
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n")
    print(f"{args.model} on {run['runs'][-1]['device']} -> {out_dir}")

    for cond, task, path, todo, total in plan:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            for call in tqdm(todo, desc=f"{cond} {task}", initial=total - len(todo), total=total):
                answer = model.answer([image_dir / im for im in call["images"]], call["prompt"])
                f.write(json.dumps({**call, **answer}, ensure_ascii=False) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
