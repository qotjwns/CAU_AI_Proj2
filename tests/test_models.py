"""
    src/models.py without any model weights: the registry, each family's input format, and that MIHBench inference
    and the training collator build the same inputs. Only the Qwen2.5-VL-3B processor is loaded (if present).
    .venv/bin/python -m unittest tests.test_models -v
"""
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cf_data  # noqa: E402
import models  # noqa: E402
from prompts import PROMPTS, fill  # noqa: E402

QWEN_DIR = ROOT / "models" / "Qwen2.5-VL-3B-Instruct"
SPLIT = cf_data.SPLITS / "pilot_train.jsonl"


class RecProcessor:
    """Records its calls (LLaVA's processor is not needed)."""

    def __init__(self):
        self.calls, self.tokenizer = [], types.SimpleNamespace()

    def __call__(self, **kw):
        self.calls.append(kw)
        return {}


class RecTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kw):
        self.calls.append((texts, kw))
        return {"input_ids": torch.zeros(len(texts), 1, dtype=torch.long)}

    def convert_tokens_to_ids(self, token):
        return 7


class Registry(unittest.TestCase):
    def test_every_family_is_complete(self):
        for name, (folder, family) in models.MODELS.items():
            self.assertIn(family.FAMILY, PROMPTS, name)
            self.assertEqual(family.Inputs.FAMILY, family.FAMILY, name)
            self.assertTrue(family.VISION and family.LEGACY_PROMPT, name)
            for method in ("answer", "for_training", "answer_outputs", "train_mode", "hidden_states", "lm_head"):
                self.assertTrue(callable(getattr(family, method)), (name, method))


@unittest.skipUnless(SPLIT.exists(), "data/coco_existence not built")
class Formats(unittest.TestCase):
    def setUp(self):
        group = cf_data.GroupDataset("pilot_train")[0]
        self.paths = [cf_data.image_paths(c) for c in group[:2]]
        self.questions = [c["question"] for c in group[:2]]

    def test_llava_text(self):
        proc = RecProcessor()
        inputs = models.LlavaOVInputs(proc)
        prompt = fill(inputs.template, self.questions[0])
        inputs.encode([prompt], self.paths[:1], ["Yes<|im_end|>"])
        call = proc.calls[0]
        self.assertEqual(call["text"], ["<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                                        "<image>\n<image>\n<image>\n\n" + prompt
                                        + "<|im_end|>\n<|im_start|>assistant\nYes<|im_end|>"])
        self.assertEqual([len(x) for x in call["images"]], [3])  # nested: one multi-image sample
        self.assertEqual(call["images"][0][0].size, (672, 672))

    def test_internvl_text_and_tiles(self):
        tok = RecTokenizer()
        inputs = models.InternVLInputs(tok, ("<before>", "<after>"), num_image_token=2)
        batch = inputs.encode(["P1", "P2"], self.paths, ["A1", "A2"])
        tiles = [len(inputs.tiles(p)) for p in self.paths]
        texts = tok.calls[0][0]
        for text, prompt, after, n in zip(texts, ["P1", "P2"], ["A1", "A2"], tiles):
            self.assertEqual(text, "<before><img>" + "<IMG_CONTEXT>" * (2 * n) + "</img>\n" + prompt + "<after>" + after)
        self.assertEqual(tuple(batch["pixel_values"].shape), (sum(tiles), 3, 448, 448))


@unittest.skipUnless(QWEN_DIR.is_dir() and SPLIT.exists(), "Qwen2.5-VL-3B processor or data missing")
class QwenSame(unittest.TestCase):
    """The official run's preprocessing (chat template, then the processor) is what both inference and the
    training collator feed the model."""

    @classmethod
    def setUpClass(cls):
        cls.inputs = models.QwenVL.load_inputs(QWEN_DIR)
        cls.group = cf_data.GroupDataset("pilot_train")[0]

    def test_inference_inputs_are_the_official_preprocessing(self):
        paths, prompt = cf_data.image_paths(self.group[0]), fill(self.inputs.template, self.group[0]["question"])
        proc = self.inputs.processor
        content = [{"type": "image"} for _ in paths] + [{"type": "text", "text": prompt}]
        text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        want = proc(text=[text], images=models.open_images(paths, 672), return_tensors="pt")
        got = self.inputs.encode([prompt], [paths])
        self.assertEqual(set(want), set(got))
        for k in want:
            self.assertTrue(torch.equal(want[k], got[k]), k)

    def test_training_batch_is_the_inference_prompt_plus_the_answer(self):
        collate = cf_data.Collator(self.inputs)
        batch = collate([self.group])
        for i, case in enumerate(self.group):
            alone = self.inputs.encode([collate.prompt(case)], [cf_data.image_paths(case)])["input_ids"][0]
            n = len(alone)
            self.assertTrue(torch.equal(batch["input_ids"][i, :n], alone), case["case_id"])  # prompt part identical
            self.assertEqual(batch["answer_index"][i].item(), n - 1)  # logits at the prompt's end predict the answer
            want = collate.yes_id if case["gt"] == "Yes" else collate.no_id
            self.assertEqual(batch["labels"][i, n].item(), want)


if __name__ == "__main__":
    unittest.main()
