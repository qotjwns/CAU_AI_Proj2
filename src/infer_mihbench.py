#!/usr/bin/env python3
"""
    Run a VLM on MIHBench and score it. Each model follows its own official MIHBench script in pgtrece/DAB
    (image size, prompt, decoding); see the class docstrings.
      results/mihbench/vanilla/<model>/<task>.jsonl   one answer per line; re-running skips answered questions
      results/mihbench/vanilla/<model>/metrics.json   accuracy, precision, recall, F1, yes-ratio and the run
                                                      environment (model commit, 4-bit, versions, GPU, date) per task
    python src/infer_mihbench.py --model qwen2.5-vl-3b                            # all 5 tasks
    python src/infer_mihbench.py --model internvl2.5-4b --task count --limit 20  # quick check
    Models (same names as scripts/download_models.py): qwen2.5-vl-3b, qwen2.5-vl-7b, internvl2.5-4b, llava-ov-7b
"""
import argparse
import datetime
import json
import platform
from importlib import metadata
from pathlib import Path

import torch
import torchvision.transforms as T
import transformers
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from mihbench import TASKS, load_questions, parse_yes_no, score

PROJECT = Path(__file__).resolve().parents[1]


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def open_images(paths, size=None):
    images = [Image.open(p).convert("RGB") for p in paths]
    return [im.resize((size, size)) for im in images] if size else images


class QwenVL:
    """Qwen2.5-VL/qwen2.5vl_eval.py: images resized to 672x672 (576 tokens each), the question as is,
    greedy, at most 128 new tokens. The official run loads 4-bit weights (--load-4bit)."""

    def __init__(self, path, load_4bit=False):
        from transformers import Qwen2_5_VLForConditionalGeneration
        kwargs = {"dtype": torch.bfloat16}
        if load_4bit:  # needs CUDA and `pip install bitsandbytes`
            from transformers import BitsAndBytesConfig
            kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map="auto")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, **kwargs).eval()
        if not load_4bit:
            self.model.to(pick_device())
        self.processor = AutoProcessor.from_pretrained(path, use_fast=False)  # the official run's preprocessing

    @torch.inference_mode()
    def answer(self, image_paths, question):
        images = open_images(image_paths, 672)
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": question}]
        prompt = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                    tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=None)
        return self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


def internvl_tiles(image, max_num=12, size=448):
    """dynamic_preprocess from the InternVL2.5 model card: cut the image into the grid of at most max_num
    size x size tiles whose aspect ratio is closest to the image's, then add a thumbnail of the whole image."""
    w, h = image.size
    grids = sorted({(i, j) for n in range(1, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                    if i * j <= max_num}, key=lambda g: g[0] * g[1])
    best, best_diff = (1, 1), float("inf")
    for g in grids:
        diff = abs(w / h - g[0] / g[1])
        if diff < best_diff:
            best, best_diff = g, diff
        elif diff == best_diff and w * h > 0.5 * size * size * g[0] * g[1]:
            best = g
    cols, rows = best
    resized = image.resize((size * cols, size * rows))
    tiles = [resized.crop((c * size, r * size, (c + 1) * size, (r + 1) * size))
             for r in range(rows) for c in range(cols)]
    return tiles + [image.resize((size, size))] if len(tiles) > 1 else tiles


class InternVL:
    """InternVL2.5/internvl2.5_eval.py: each image (not resized) cut into up to 12 448px tiles plus a thumbnail,
    the tiles of all images given as one '<image>', ' Please answer with yes or no.' appended,
    greedy, at most 1024 new tokens. The official run uses eager attention, which builds a
    tokens x tokens matrix per layer and runs out of memory on 24 GB GPUs; sdpa=True computes the same
    attention without it."""

    def __init__(self, path, load_4bit=False, sdpa=False):
        from transformers import AutoModel, AutoTokenizer
        self.model = AutoModel.from_pretrained(path, dtype=torch.bfloat16, use_flash_attn=False,
                                               trust_remote_code=True).eval().to(pick_device())
        if sdpa:  # the remote code hard-wires eager into the language model's config
            self.model.language_model.config._attn_implementation = "sdpa"
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        self.transform = T.Compose([T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(),
                                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    @torch.inference_mode()
    def answer(self, image_paths, question):
        tiles = [t for im in open_images(image_paths) for t in internvl_tiles(im)]
        pixel_values = torch.stack([self.transform(t) for t in tiles]).to(self.model.device, torch.bfloat16)
        prompt = "<image>\n" + question + " Please answer with yes or no."
        return self.model.chat(self.tokenizer, pixel_values, prompt,
                               dict(max_new_tokens=1024, do_sample=False)).strip()


class LlavaOV:
    """LLaVA-NeXT-Interleave/llava_next_eval.py (the paper's LLaVA): images resized to 672x672 and each
    encoded once at the base resolution, '<image>\\n' per image then '\\n' + the question, greedy,
    at most 1024 new tokens, fp16."""

    def __init__(self, path, load_4bit=False):
        from transformers import LlavaOnevisionForConditionalGeneration
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(path, dtype=torch.float16).eval()
        self.model.to(pick_device())
        self.processor = AutoProcessor.from_pretrained(path, use_fast=False)

    @torch.inference_mode()
    def answer(self, image_paths, question):
        images = open_images(image_paths, 672)
        prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                  + "<image>\n" * len(images) + "\n" + question + "<|im_end|>\n<|im_start|>assistant\n")
        # images as one nested list = one multi-image sample: base resolution only (729 + 1 tokens per image).
        # A flat list would be treated as separate single images and cut into anyres tiles (~3,700 tokens each).
        inputs = self.processor(text=[prompt], images=[images], return_tensors="pt")
        inputs = inputs.to(self.model.device, torch.float16)  # casts only the float tensors (pixels)
        out = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False,
                                  pad_token_id=self.processor.tokenizer.eos_token_id)
        return self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()


MODELS = {  # name -> (folder under models/, wrapper)
    "qwen2.5-vl-3b": ("Qwen2.5-VL-3B-Instruct", QwenVL),
    "qwen2.5-vl-7b": ("Qwen2.5-VL-7B-Instruct", QwenVL),
    "internvl2.5-4b": ("InternVL2_5-4B", InternVL),
    "llava-ov-7b": ("llava-onevision-qwen2-7b-ov-hf", LlavaOV),
}


def read_answers(path):
    """Answers saved so far, by question_id. Drops a line cut off by an interrupted run."""
    if not path.exists():
        return {}
    rows, lines = {}, path.read_text().splitlines()
    for line in lines:
        try:
            row = json.loads(line)
            rows[row["question_id"]] = row
        except (json.JSONDecodeError, KeyError):
            pass
    if len(rows) != len(lines):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
    return rows


def run_info(name, model_dir, model, load_4bit, sdpa=False):
    """The environment that produced the answers: model files, precision, packages, hardware, date."""
    meta = model_dir / ".cache" / "huggingface" / "download" / "config.json.metadata"  # line 1: commit
    device = model.model.device
    info = {
        "method": "vanilla",
        "model": name,
        "model_commit": meta.read_text().splitlines()[0] if meta.exists() else None,
        "load_4bit": load_4bit,
        "attn": "sdpa" if sdpa else "official",
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if load_4bit:
        info["bitsandbytes"] = metadata.version("bitsandbytes")
    return info


def run_task(model, task, out_dir, limit):
    questions = load_questions(task)[:limit]
    path = out_dir / f"{task}.jsonl"
    answers = read_answers(path)
    todo = [q for q in questions if q["question_id"] not in answers]
    with open(path, "a") as f:
        for q in tqdm(todo, desc=task, initial=len(questions) - len(todo), total=len(questions)):
            text = model.answer(q["image_paths"], q["text"])
            row = {"question_id": q["question_id"], "question": q["text"], "image_list": q["image_list"],
                   "text": text, "pred": parse_yes_no(text), "label": q["label"], "injected": q.get("injected")}
            f.write(json.dumps(row) + "\n")
            f.flush()
            answers[q["question_id"]] = row
    return score([answers[q["question_id"]] for q in questions]), len(todo)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--task", nargs="+", default=["all"], choices=[*TASKS, "all"])
    ap.add_argument("--limit", type=int, help="only the first N questions of each task")
    ap.add_argument("--load-4bit", action="store_true", help="4-bit weights like the official Qwen run (CUDA only)")
    ap.add_argument("--sdpa", action="store_true",
                    help="InternVL only: sdpa attention instead of the official eager (needed on 24 GB GPUs)")
    args = ap.parse_args()
    tasks = list(TASKS) if "all" in args.task else list(dict.fromkeys(args.task))
    folder, wrapper = MODELS[args.model]
    if args.load_4bit and wrapper is not QwenVL:
        ap.error("--load-4bit is only for the Qwen models (the other official scripts run full precision)")
    if args.sdpa and wrapper is not InternVL:
        ap.error("--sdpa is only for InternVL (the other models already use sdpa)")

    model_dir = PROJECT / "models" / folder
    if not model_dir.is_dir():
        raise SystemExit(f"{model_dir} not found: python scripts/download_models.py --model {args.model}")
    suffix = ("-4bit" if args.load_4bit else "") + ("-sdpa" if args.sdpa else "")
    out_dir = PROJECT / "results" / "mihbench" / "vanilla" / (args.model + suffix)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = InternVL(model_dir, sdpa=True) if args.sdpa else wrapper(model_dir, args.load_4bit)
    info = run_info(args.model, model_dir, model, args.load_4bit, args.sdpa)
    print(f"{args.model} on {info['device']}")
    metrics_path = out_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    for task in tasks:
        result, n_new = run_task(model, task, out_dir, args.limit)
        old = metrics.get(task, {}).get("run")
        # Only re-scored (no new answers): keep the environment that produced the answers.
        metrics[task] = {**result, "run": info if n_new or old is None else old}
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"\n{'task':<24}{'n':>5}{'acc':>8}{'prec':>8}{'rec':>8}{'f1':>8}{'yes':>8}")
    for task in tasks:
        m = metrics[task]
        print(f"{task:<24}{m['n']:>5}" + "".join(f"{m[k]:>8.3f}" for k in
                                                 ("accuracy", "precision", "recall", "f1", "yes_ratio")))
    print(f"\nAnswers and metrics: {out_dir}")


if __name__ == "__main__":
    main()
