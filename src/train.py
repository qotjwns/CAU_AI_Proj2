#!/usr/bin/env python3
"""
    LoRA training on the COCO existence four-case groups (src/cf_data.py): reads the options, puts the pieces
    together and records the run. The pieces:
      src/objectives.py  the methods (sft, group-sft, cf-rank, cf-invariant, cf-metric) and their loss terms
      src/runner.py      the optimizer loop (gradient accumulation) and the dev loop
      src/models.py      per model family: loading, inputs (shared with MIHBench inference), LoRA targets, the
                         forward pass to the answer position
      src/cf_data.py     the groups and the collator
      src/metrics.py     dev metrics from the answer margin
      src/runinfo.py     device, model commit and versions, recorded in args.json
    Every method uses the same data, model and number of cases per batch (--batch-size); one epoch is 1,170
    optimizer steps at --batch-size 8 for every method. The scheduler is planned for exactly the steps taken
    (runner.planned_steps), and the run stops with an error if they differ.
    Prompt: the 3 images, then the model family's prompt from configs/prompts.yaml filled with
    "Is there a/an <class> in all 3 images?", built as in its MIHBench inference.
    Loss weights and margins are placeholders until tuned: --lambda-flip 1, --lambda-stay 1, --gamma 2,
    --epsilon 0.5, --alpha 1, --beta 1, --proj-dim 128, --bank-size 256.
      runs/<method>/<model>[-<tag>]/adapter/    LoRA weights (peft), saved after every epoch
      runs/<method>/<model>[-<tag>]/args.json   settings and environment
      runs/<method>/<model>[-<tag>]/log.jsonl   train loss (and each term) every --log-every steps, dev metrics
                                                (src/metrics.py) and dev terms before training and after every epoch
      runs/cf-metric/<model>[-<tag>]/projection_head.pt   the projection head, saved with the adapter (not needed
                                                          for inference)
    MIHBench with the trained adapter (results/mihbench/<method>/<model>[-<tag>]/, then summary and figures):
    python src/infer_mihbench.py --adapter runs/<method>/<model>[-<tag>]
    python src/eval_mihbench.py --method <method>
    Pilot (468 groups), per model. InternVL (~10k tokens per case) takes 4 cases per batch and 2 batches per
    optimizer step, the same 8 cases per step:
    python src/train.py --model qwen2.5-vl-3b --method sft --train pilot_train --tag pilot
    python src/train.py --model qwen2.5-vl-3b --method cf-invariant --train pilot_train --tag pilot
    python src/train.py --model qwen2.5-vl-7b --method sft --train pilot_train --tag pilot
    python src/train.py --model llava-ov-7b --method sft --train pilot_train --tag pilot
    python src/train.py --model internvl2.5-4b --method sft --train pilot_train --batch-size 4 --grad-accum 2 --tag pilot
    python src/train.py --model qwen2.5-vl-3b --method sft --max-steps 2 --eval-groups 2 --tag smoke  # quick check
    Full training: --train train (the default) instead of --train pilot_train.
"""
import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from cf_data import CaseDataset, Collator, GroupDataset
from models import LORA_TARGETS, MODELS
from objectives import METHODS, Objective
from runinfo import environment, model_commit, pick_device
from runner import evaluate, fit, items_per_batch, planned_steps

PROJECT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--train", default="train", choices=["train", "pilot_train"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, help="stop after this many optimizer steps")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="cases per batch, the same for every method (a multiple of 4: group methods take whole groups)")
    ap.add_argument("--grad-accum", type=int, default=1, help="batches per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.03, help="share of the steps")
    ap.add_argument("--eval-groups", type=int, help="only this many dev groups (a fixed random sample)")
    ap.add_argument("--no-dev-before", action="store_true",
                    help="skip the dev evaluation before training (the same for every method of a model)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", help="appended to the run folder name")
    terms = ap.add_argument_group("loss terms (src/objectives.py; placeholders until tuned)")
    terms.add_argument("--lambda-flip", type=float, default=1.0, help="weight of Flip (cf-rank, cf-invariant)")
    terms.add_argument("--lambda-stay", type=float, default=1.0, help="weight of Stay (cf-invariant)")
    terms.add_argument("--gamma", type=float, default=2.0, help="Flip margin, in logit units")
    terms.add_argument("--epsilon", type=float, default=0.5, help="Stay tolerance, in logit units")
    terms.add_argument("--alpha", type=float, default=1.0, help="weight of Align (cf-metric)")
    terms.add_argument("--beta", type=float, default=1.0, help="weight of Uniform (cf-metric)")
    terms.add_argument("--proj-dim", type=int, default=128, help="projection head output size (cf-metric)")
    terms.add_argument("--bank-size", type=int, default=256,
                       help="training features kept for Uniform (cf-metric); 0: the batch alone")
    args = ap.parse_args()
    if args.batch_size % 4:
        ap.error("--batch-size must be a multiple of 4, so every method can use the same batch size")
    batching, loss_terms = METHODS[args.method]

    from peft import LoraConfig, __version__ as peft_version, get_peft_model
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    folder, family = MODELS[args.model]
    model_dir = PROJECT / "models" / folder
    run_dir = PROJECT / "runs" / args.method / (args.model + (f"-{args.tag}" if args.tag else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(pick_device())

    model, lm, inputs = family.for_training(model_dir)
    collate = Collator(inputs)
    model.to(device)
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS))
    hidden_size = family.lm_head(model.get_base_model()).in_features
    objective = Objective(loss_terms, collate, hidden_size, args).to(device)

    def forward(batch):
        return family.answer_outputs(model, batch, device)

    def run_dev():
        model.eval()
        metrics = evaluate(dev_loader, collate, objective, forward)
        family.train_mode(model)
        return metrics
    trainable = sum(p.numel() for p in [*model.parameters(), *objective.parameters()] if p.requires_grad)

    # sft: items are cases; group methods: items are groups of 4 cases. Dev is always read by group (group metrics).
    train = CaseDataset(args.train) if batching == "case" else GroupDataset(args.train)
    dev = GroupDataset("dev")
    if args.eval_groups:
        dev = Subset(dev, sorted(random.Random(args.seed).sample(range(len(dev)), args.eval_groups)))
    loader_kw = {"collate_fn": collate, "num_workers": args.workers, "persistent_workers": args.workers > 0}
    train_loader = DataLoader(train, batch_size=items_per_batch(batching, args.batch_size), shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), **loader_kw)
    dev_loader = DataLoader(dev, batch_size=items_per_batch("group", args.batch_size), shuffle=False, **loader_kw)
    steps = planned_steps(len(train_loader), args.grad_accum, args.epochs, args.max_steps)

    params = [p for p in [*model.parameters(), *objective.parameters()] if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, round(args.warmup * steps), steps)
    info = {**vars(args), "batching": batching, "loss": ["ce", *loss_terms], "train_items": len(train),
            "dev_groups": len(dev), "steps": steps,
            "trainable_params": trainable, "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": LORA_TARGETS},
            "model_commit": model_commit(model_dir), "prompt": collate.template, "dtype": "bfloat16",
            **environment(device), "peft": peft_version}
    (run_dir / "args.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"{run_dir.relative_to(PROJECT)}: {len(train)} train {batching}s, {steps} steps, "
          f"{trainable / 1e6:.1f}M trainable parameters, {device}")

    log = open(run_dir / "log.jsonl", "w")
    def write(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    if not args.no_dev_before:
        dev_metrics = run_dev()
        write({"step": 0, "epoch": 0, "dev": dev_metrics})
        print(f"dev before training: {dev_metrics}")

    family.train_mode(model)
    objective.train()
    window, part_window = [], []
    bar = tqdm(total=steps, desc=args.method)

    def loss_of(batch):
        loss, parts = objective(*forward(batch), batch)
        part_window.append(parts)
        return loss

    def on_step(step, epoch, loss, grad_norm):
        window.append(loss)
        bar.update()
        bar.set_postfix(loss=f"{sum(window) / len(window):.4f}")
        if step % args.log_every == 0 or step == steps:
            parts = {k: round(sum(p[k] for p in part_window) / len(part_window), 5) for k in part_window[0]}
            write({"step": step, "epoch": epoch, "loss": round(sum(window) / len(window), 5), "terms": parts,
                   "grad_norm": round(grad_norm, 4), "lr": sched.get_last_lr()[0]})
            window.clear()
            part_window.clear()

    def on_epoch(step, epoch):
        model.save_pretrained(run_dir / "adapter")
        if objective.head is not None:
            torch.save(objective.head.state_dict(), run_dir / "projection_head.pt")
        dev_metrics = run_dev()
        write({"step": step, "epoch": epoch, "dev": dev_metrics})
        print(f"\ndev after epoch {epoch}: {dev_metrics}")

    taken = fit(params, opt, sched, train_loader, loss_of, args.epochs, steps, args.grad_accum, on_step, on_epoch)
    bar.close()
    log.close()
    if taken != steps or sched.last_epoch != steps:
        raise RuntimeError(f"took {taken} optimizer steps and {sched.last_epoch} scheduler steps, planned {steps}")
    print(f"\n{run_dir / 'adapter'}\nMIHBench: python src/infer_mihbench.py --adapter {run_dir.relative_to(PROJECT)}"
          f"\n          python src/eval_mihbench.py --method {args.method}")


if __name__ == "__main__":
    main()
