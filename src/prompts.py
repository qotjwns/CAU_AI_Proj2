"""
    The text prompt of each model family, read once from configs/prompts.yaml (edit the prompts there).
      PROMPTS["qwen2.5-vl"]   "{question} Please answer with Yes or No."
      fill(template, q)       the template with {question} replaced by q
    Used by src/infer_mihbench.py and src/cf_data.py, so evaluation and training prompts cannot drift apart.
"""
from pathlib import Path

import yaml

PATH = Path(__file__).resolve().parents[1] / "configs" / "prompts.yaml"
FAMILIES = ("qwen2.5-vl", "internvl2.5", "llava-ov")
SLOT = "{question}"


def load(path=PATH):
    prompts = yaml.safe_load(path.read_text()) or {}
    problems = [f"{f}: missing" for f in FAMILIES if f not in prompts]
    problems += [f"{f}: needs {SLOT} exactly once" for f, t in prompts.items()
                 if not isinstance(t, str) or t.count(SLOT) != 1]
    if problems:
        raise SystemExit(f"{path}: " + "; ".join(problems))
    return prompts


def fill(template, question):
    return template.replace(SLOT, question)


PROMPTS = load()
