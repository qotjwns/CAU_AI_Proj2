"""
    MIHBench existence questions (data/mihbench/questions/<task>.jsonl, scripts/download_mihbench.py).
      load_questions(path)    the questions of one task file, in file order
      question_object(text)   the class a question asks about: "Is there a cat in all 3 images?" -> "cat".
                              Also reads the 86 questions written "Is there an X in the imange? in all 3 images?"
      parse_yes_no(text)      the official answer rule (eval.py in pgtrece/DAB): "no" if the first sentence has
                              the word No, no or not, else "yes"
"""
import json
import re
from pathlib import Path

OBJECT = re.compile(r"Is there an? (.+?)(?: in the imange\?)? in all \d+ images\?")


def load_questions(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def question_object(text):
    match = OBJECT.fullmatch(text.strip())
    if not match:
        raise ValueError(f"not an existence question: {text!r}")
    return match.group(1)


def parse_yes_no(text):
    words = text.split(".")[0].replace(",", "").split(" ")
    return "no" if {"No", "no", "not"} & set(words) else "yes"
