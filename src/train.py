#!/usr/bin/env python3
"""
    LoRA training on the COCO existence four-case groups (src/cf_data.py). Every method uses the same data,
    batches and model; only the loss differs. Implemented so far:
      sft   cross-entropy on each case's answer ("Yes" / "No") and the <|im_end|> after it
    Prompt: the 3 images, then the model family's prompt from configs/prompts.yaml filled with
    "Is there a/an <class> in all 3 images?", built as in its MIHBench inference (src/cf_data.py has one collator
    per model family).
    Batches hold whole groups (--groups-per-step groups = 4x cases), in a new order every epoch.
    LoRA (r 16, alpha 32, dropout 0.05) on the language model's attention and MLP projections; the vision encoder
    and the projector stay frozen (and in eval mode). bf16 weights for every model (LLaVA-OV runs in fp16 at
    inference; fp16 training without loss scaling is unstable), gradient checkpointing on the language model.
    Only the hidden states at the answer position and the one after it go through lm_head.
    InternVL is remote code: its <IMG_CONTEXT> embeddings are replaced by the tile features here, as its generate()
    does, and its language model runs with sdpa instead of the hard-wired eager attention (~10k tokens per case).
      runs/<method>/<model>[-<tag>]/adapter/    LoRA weights (peft), saved after every epoch
      runs/<method>/<model>[-<tag>]/args.json   settings and environment
      runs/<method>/<model>[-<tag>]/log.jsonl   train loss every --log-every steps, dev metrics before training
                                                and after every epoch
    Dev metrics, from the margin m = logit(Yes) - logit(No) where the answer starts:
      acc        m > 0 exactly on the Yes cases          yes_ratio  share of cases with m > 0
      group_acc  all 4 cases of a group right            flip_rank  m_A > m_B and m_C > m_D, per flip pair
      stay_gap   mean |m_A - m_C| and |m_B - m_D|: 0 when the answer ignores the swapped P image
    Pilot (468 groups), per model; InternVL takes 1 group per batch and 2 batches per step (the same 8 cases):
    python src/train.py --model qwen2.5-vl-3b --train pilot_train --tag pilot
    python src/train.py --model qwen2.5-vl-7b --train pilot_train --tag pilot
    python src/train.py --model llava-ov-7b --train pilot_train --tag pilot
    python src/train.py --model internvl2.5-4b --train pilot_train --groups-per-step 1 --grad-accum 2 --tag pilot
    python src/train.py --model qwen2.5-vl-3b --max-steps 2 --eval-groups 2 --tag smoke  # quick check
    Full training: --train train (the default) instead of --train pilot_train.
"""
import argparse
import datetime
import json
import platform
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import (AutoModel, AutoProcessor, AutoTokenizer, LlavaOnevisionForConditionalGeneration,
                          Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup)

from cf_data import FLIP_PAIRS, STAY_PAIRS, GroupDataset, InternVLCollator, LlavaOVCollator, QwenCollator
from infer_mihbench import MODELS, pick_device

PROJECT = Path(__file__).resolve().parents[1]
# language-model layers only: model.language_model.layers.N (Qwen2.5-VL, LLaVA-OV),
# language_model.model.layers.N (InternVL); vision layers never match
LORA_TARGETS = r".*language_model\.(model\.)?layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)"
VISION = ("model.visual", "model.vision_tower", "vision_model")  # Qwen2.5-VL, LLaVA-OV, InternVL
NOT_INPUTS = ("labels", "answer_index", "target")


def internvl_chat(model):
    """The text InternVL's model.chat() puts around a query, split at the query: (before, after)."""
    conv = sys.modules[type(model).__module__].get_conv_template(model.template)
    conv.system_message = model.system_message
    conv.append_message(conv.roles[0], "<<QUERY>>")
    conv.append_message(conv.roles[1], None)
    return tuple(conv.get_prompt().split("<<QUERY>>"))


def load_model(name, model_dir):
    """(model, its language model, collator), bf16 on the CPU. Gradient checkpointing goes on the second."""
    if name.startswith("qwen"):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16)
        return model, model, QwenCollator(AutoProcessor.from_pretrained(model_dir, use_fast=False))
    if name.startswith("llava"):
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_dir, dtype=torch.bfloat16)
        return model, model, LlavaOVCollator(AutoProcessor.from_pretrained(model_dir, use_fast=False))
    model = AutoModel.from_pretrained(model_dir, dtype=torch.bfloat16, use_flash_attn=False, trust_remote_code=True)
    model.language_model.config._attn_implementation = "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    collate = InternVLCollator(tokenizer, internvl_chat(model), model.num_image_token)
    model.img_context_token_id = collate.context_id
    return model, model.language_model, collate


def train_mode(model):
    """Training mode (LoRA dropout on) except for the frozen vision encoder."""
    model.train()
    net = model.get_base_model()
    for path in VISION:
        try:
            net.get_submodule(path).eval()
        except AttributeError:
            pass


def answer_logits(model, batch, device):
    """Logits (cases, 2, vocab) at the answer position (predicting Yes / No) and the next one (predicting
    <|im_end|>), in float32."""
    net = model.get_base_model()
    if hasattr(net, "extract_feature"):  # InternVL: tile features into the <IMG_CONTEXT> embeddings
        lm, ids = net.language_model, batch["input_ids"].to(device)
        vit = net.extract_feature(batch["pixel_values"].to(device, net.dtype))
        emb = lm.get_input_embeddings()(ids)
        emb = emb.masked_scatter((ids == net.img_context_token_id).unsqueeze(-1).expand_as(emb), vit.to(emb.dtype))
        h = lm.model(inputs_embeds=emb, attention_mask=batch["attention_mask"].to(device),
                     use_cache=False).last_hidden_state
        head = lm.lm_head
    else:  # Qwen2.5-VL, LLaVA-OV: the processor's outputs are the inputs
        inputs = {k: v.to(device, net.dtype) if v.is_floating_point() else v.to(device)
                  for k, v in batch.items() if k not in NOT_INPUTS}
        h = net.model(**inputs, use_cache=False).last_hidden_state
        head = net.lm_head
    rows = torch.arange(h.shape[0], device=h.device)
    idx = batch["answer_index"].to(h.device)
    return head(torch.stack([h[rows, idx], h[rows, idx + 1]], 1)).float()


def sft_loss(logits, batch):
    rows, idx = torch.arange(len(logits)), batch["answer_index"]
    targets = torch.stack([batch["labels"][rows, idx + 1], batch["labels"][rows, idx + 2]], 1).to(logits.device)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


LOSSES = {"sft": sft_loss}


@torch.no_grad()
def evaluate(model, loader, collate, device):
    model.eval()
    margins, losses = [], []
    for batch in tqdm(loader, desc="dev", leave=False):
        logits = answer_logits(model, batch, device)
        margins.append((logits[:, 0, collate.yes_id] - logits[:, 0, collate.no_id]).cpu())
        losses.append(sft_loss(logits, batch).item())
    train_mode(model)
    m = torch.cat(margins).view(-1, 4)  # groups x cases A, B, C, D
    right = (m > 0) == torch.tensor([True, False, True, False])
    return {"groups": len(m), "ce": round(sum(losses) / len(losses), 4),
            "acc": round(right.float().mean().item(), 4),
            "yes_ratio": round((m > 0).float().mean().item(), 4),
            "group_acc": round(right.all(1).float().mean().item(), 4),
            "flip_rank": round(torch.stack([m[:, i] > m[:, j] for i, j in FLIP_PAIRS]).float().mean().item(), 4),
            "stay_gap": round(torch.stack([(m[:, i] - m[:, j]).abs() for i, j in STAY_PAIRS]).mean().item(), 3)}


def environment(model_dir, device, template):
    meta = model_dir / ".cache" / "huggingface" / "download" / "config.json.metadata"  # line 1: commit
    return {"model_commit": meta.read_text().splitlines()[0] if meta.exists() else None,
            "prompt": template, "dtype": "bfloat16",
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": __import__("peft").__version__,
            "date": datetime.datetime.now().isoformat(timespec="seconds")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--method", default="sft", choices=LOSSES)
    ap.add_argument("--train", default="train", choices=["train", "pilot_train"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, help="stop after this many optimizer steps")
    ap.add_argument("--groups-per-step", type=int, default=2, help="groups per batch (4 cases each)")
    ap.add_argument("--grad-accum", type=int, default=1, help="batches per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.03, help="share of the steps")
    ap.add_argument("--eval-groups", type=int, help="only this many dev groups (a fixed random sample)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", help="appended to the run folder name")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_dir = PROJECT / "models" / MODELS[args.model][0]
    run_dir = PROJECT / "runs" / args.method / (args.model + (f"-{args.tag}" if args.tag else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(pick_device())

    model, lm, collate = load_model(args.model, model_dir)
    model.to(device)
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train = GroupDataset(args.train)
    dev = GroupDataset("dev")
    if args.eval_groups:
        dev = Subset(dev, sorted(random.Random(args.seed).sample(range(len(dev)), args.eval_groups)))
    loader_kw = {"batch_size": args.groups_per_step, "collate_fn": collate, "num_workers": args.workers,
                 "persistent_workers": args.workers > 0}
    train_loader = DataLoader(train, shuffle=True, generator=torch.Generator().manual_seed(args.seed), **loader_kw)
    dev_loader = DataLoader(dev, shuffle=False, **loader_kw)
    steps = args.epochs * len(train_loader) // args.grad_accum
    if args.max_steps:
        steps = min(steps, args.max_steps)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, round(args.warmup * steps), steps)
    info = {**vars(args), "train_groups": len(train), "dev_groups": len(dev), "steps": steps,
            "trainable_params": trainable, "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": LORA_TARGETS},
            **environment(model_dir, device, collate.template)}
    (run_dir / "args.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"{run_dir.relative_to(PROJECT)}: {len(train)} train groups, {steps} steps, "
          f"{trainable / 1e6:.1f}M trainable parameters, {device}")

    log = open(run_dir / "log.jsonl", "w")
    def write(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    dev_metrics = evaluate(model, dev_loader, collate, device)
    write({"step": 0, "epoch": 0, "dev": dev_metrics})
    print(f"dev before training: {dev_metrics}")

    train_mode(model)
    step, window = 0, []
    bar = tqdm(total=steps, desc=args.method)
    for epoch in range(1, args.epochs + 1):
        for i, batch in enumerate(train_loader):
            loss = LOSSES[args.method](answer_logits(model, batch, device), batch)
            (loss / args.grad_accum).backward()
            window.append(loss.item())
            if (i + 1) % args.grad_accum:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            bar.update()
            bar.set_postfix(loss=f"{sum(window) / len(window):.4f}")
            if step % args.log_every == 0 or step == steps:
                write({"step": step, "epoch": epoch, "loss": round(sum(window) / len(window), 5),
                       "grad_norm": round(grad_norm, 4), "lr": sched.get_last_lr()[0]})
                window = []
            if step >= steps:
                break
        model.save_pretrained(run_dir / "adapter")
        dev_metrics = evaluate(model, dev_loader, collate, device)
        write({"step": step, "epoch": epoch, "dev": dev_metrics})
        print(f"\ndev after epoch {epoch}: {dev_metrics}")
        if step >= steps:
            break
    bar.close()
    log.close()
    print(f"\n{run_dir / 'adapter'}")


if __name__ == "__main__":
    main()
