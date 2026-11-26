#!/usr/bin/env python3
"""
    Shared training data: the COCO existence four-case groups (src/build_coco_existence.py) as model inputs.
    Every training method reads its data here, so all of them see the same groups, prompt and answer tokens.
    A group is one question about one class over 4 cases of 3 images, kept in the order A, B, C, D:
      A = P1 P2 P3 -> Yes    B = P1 P2 N1 -> No    C = P1 P4 P3 -> Yes    D = P1 P4 N1 -> No
      FLIP_PAIRS  A-B, C-D: only the flip position changes (P3 <-> N1) and the answer flips
      STAY_PAIRS  A-C, B-D: only the stay position changes (P2 <-> P4) and the answer stays
    The flip and stay positions rotate over the 6 position pairs (above: stay 2nd, flip 3rd); load_groups checks
    every group against this layout.
    Collator builds the batches with the model family's Inputs (src/models.py), the same code MIHBench inference
    uses, and the family's prompt from configs/prompts.yaml, so training and evaluation inputs match; the target is
    "Yes" or "No", then <|im_end|> (all four models use Qwen tokenizers).
      CaseDataset   one case per item (SFT on shuffled cases)
      GroupDataset  one group per item, its 4 cases in order (losses over flip and stay pairs)
    In a GroupDataset batch, row 4k+j is case "ABCD"[j] of group k.
    python src/cf_data.py   # check every split, then tokenize two pilot groups (needs models/Qwen2.5-VL-3B-Instruct)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from prompts import fill

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
SPLITS = DATA / "coco_existence"
CASES = ("A", "B", "C", "D")
ANSWERS = ("Yes", "No", "Yes", "No")
FLIP_PAIRS = ((0, 1), (2, 3))  # indices into CASES
STAY_PAIRS = ((0, 2), (1, 3))


def check_group(cases):
    """Raise ValueError unless the cases are A, B, C, D of one group, laid out as in the module docstring."""
    gid = cases[0]["group_id"]
    shared = {(c["group_id"], c["question"], c["flip_position"], c["stay_position"]) for c in cases}
    if len(shared) != 1 or [c["case"] for c in cases] != list(CASES) or [c["gt"] for c in cases] != list(ANSWERS):
        raise ValueError(f"{gid}: not cases A, B, C, D of one group: {[c['case_id'] for c in cases]}")
    f, s = cases[0]["flip_position"], cases[0]["stay_position"]
    differ = lambda i, j: [k for k in range(3) if cases[i]["image_ids"][k] != cases[j]["image_ids"][k]]
    if any(differ(i, j) != [f] for i, j in FLIP_PAIRS) or any(differ(i, j) != [s] for i, j in STAY_PAIRS):
        raise ValueError(f"{gid}: images do not change only at the flip and stay positions")


def load_groups(split):
    """[[case A, B, C, D], ...] from data/coco_existence/<split>.jsonl in file order
    (split: train, dev, test or pilot_train)."""
    groups = defaultdict(list)
    with open(SPLITS / f"{split}.jsonl") as f:
        for line in f:
            case = json.loads(line)
            groups[case["group_id"]].append(case)
    for cases in groups.values():
        cases.sort(key=lambda c: c["case"])
        check_group(cases)
    return list(groups.values())


class CaseDataset(Dataset):
    """One case per item, for training on shuffled cases (SFT)."""

    def __init__(self, split):
        self.cases = [c for g in load_groups(split) for c in g]

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        return self.cases[i]


class GroupDataset(Dataset):
    """One group per item: the list of its cases A, B, C, D."""

    def __init__(self, split):
        self.groups = load_groups(split)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        return self.groups[i]


def image_paths(case):
    return [DATA / p for p in case["images"]]


class Collator:
    """Cases, or groups of cases (flattened in order), -> one batch: the model inputs (right-padded), with labels on
    the answer token and <|im_end|> only (-100 elsewhere), plus two keys that are not model inputs:
      answer_index  the position whose logits predict the answer: logits[i, answer_index[i], [yes_id, no_id]]
      target        1 for Yes, 0 for No
    inputs: the family's Inputs (src/models.py); each case's prompt is the family's template filled with its
    question, followed by its answer."""
    end_token = "<|im_end|>"

    def __init__(self, inputs):
        self.inputs, self.template = inputs, inputs.template
        tokenizer = inputs.tokenizer
        tokenizer.padding_side = "right"  # language-model position ids count from the first token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.yes_id, self.no_id, self.end_id = tokenizer.convert_tokens_to_ids(["Yes", "No", self.end_token])

    def prompt(self, case):
        return fill(self.template, case["question"])

    def __call__(self, items):
        cases = [c for x in items for c in (x if isinstance(x, list) else [x])]
        batch = self.inputs.encode([self.prompt(c) for c in cases], [image_paths(c) for c in cases],
                                   [c["gt"] + self.end_token for c in cases])
        ids, mask = batch["input_ids"], batch["attention_mask"]
        last = mask.shape[1] - 1 - mask.flip(1).argmax(1)  # <|im_end|>, whichever side is padded
        rows = torch.arange(len(cases))
        target = torch.tensor([c["gt"] == "Yes" for c in cases], dtype=torch.long)
        answer = torch.where(target.bool(), self.yes_id, self.no_id)
        if not ((ids[rows, last] == self.end_id).all() and (ids[rows, last - 1] == answer).all()):
            raise ValueError("the answer is not a single Yes / No token right before <|im_end|>")
        labels = torch.full_like(ids, -100)
        labels[rows, last - 1] = answer
        labels[rows, last] = self.end_id
        batch["labels"], batch["answer_index"], batch["target"] = labels, last - 2, target
        return batch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=PROJECT / "models" / "Qwen2.5-VL-3B-Instruct",
                   help="only its processor is loaded")
    args = p.parse_args()
    for split in ("train", "dev", "test", "pilot_train"):
        groups = load_groups(split)
        print(f"{split:<12}{len(groups):>5} groups, {4 * len(groups):>6} cases: layout checked")

    from models import QwenVL
    collate = Collator(QwenVL.load_inputs(args.model))
    data = GroupDataset("pilot_train")
    first = data[0]
    other = next(g for g in data if " " in g[0]["class_name"])  # a longer question, so row 0-3 get padded
    batch = collate([first, other])
    tok = collate.inputs.tokenizer
    print(f"\nbatch of 2 groups: input_ids {tuple(batch['input_ids'].shape)}, pixel_values "
          f"{tuple(batch['pixel_values'].shape)}, image_grid_thw {batch['image_grid_thw'][0].tolist()} x "
          f"{len(batch['image_grid_thw'])}")
    print(f"prompt (row 0): {tok.decode(batch['input_ids'][0]).split('<|vision_end|>')[-1]!r}")
    for i, c in enumerate(first + other):
        mark = ["F" if k == c["flip_position"] else "S" if k == c["stay_position"] else "-" for k in range(3)]
        print(f"  row {i}  {c['case_id']:<24} positions {''.join(mark)}  images {c['image_ids']}  "
              f"labels {tok.decode(batch['labels'][i][batch['labels'][i] != -100])!r}  "
              f"answer_index {batch['answer_index'][i].item()}  target {batch['target'][i].item()}")


if __name__ == "__main__":
    main()
