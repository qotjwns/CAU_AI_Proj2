"""
    Seen / unseen class subsets of src/eval_mihbench.py, on the real MIHBench questions with made-up answers.
    .venv/bin/python -m unittest tests.test_eval_classes -v
"""
import csv
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import eval_mihbench as ev  # noqa: E402
from mihbench import COCO_CLASSES, ROOT, TASKS, question_class  # noqa: E402

EXCLUDED = ("hair drier", "toaster")  # the classes src/coco_pools.py leaves out now
ANNOTATIONS = ROOT.parent / "coco" / "annotations" / "instances_val2014.json"


def questions(task):
    with open(ROOT / "questions" / f"{task}.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def fake_data():
    """{model: {task: answers}}: every third question answered No, the rest Yes."""
    return {"fake-model": {task: [{"question_id": q["question_id"], "question": q["text"], "label": q["label"],
                                   "text": "No" if q["question_id"] % 3 == 0 else "Yes"} for q in questions(task)]
                           for task in TASKS}}


def by_key(rows):
    return {(r["task"], r["subset"]): r for r in rows}


@unittest.skipUnless((ROOT / "questions").is_dir(), "MIHBench questions not downloaded")
class ClassSplit(unittest.TestCase):
    def test_every_question_has_its_class(self):
        for task in TASKS:
            classes = [question_class(task, q["text"]) for q in questions(task)]
            if task == "id_consistency":
                self.assertEqual(set(classes), {None})
            else:
                self.assertTrue(set(classes) <= set(COCO_CLASSES), task)

    @unittest.skipUnless(ANNOTATIONS.exists(), "instances_val2014.json not downloaded")
    def test_coco_classes_are_the_annotation_categories(self):
        cats = json.loads(ANNOTATIONS.read_text())["categories"]
        self.assertEqual(COCO_CLASSES, tuple(c["name"] for c in sorted(cats, key=lambda c: c["id"])))

    def test_same_questions_as_the_old_regex(self):
        """Regression: with classes left out, the split equals the regex it replaced."""
        old = re.compile(r"\b(?:" + "|".join(map(re.escape, EXCLUDED)) + r")s?\b")
        expected = {"existence_random": 18, "existence_popular": 20, "existence_adversarial": 20, "count": 7,
                    "id_consistency": 0}
        for task, answers in fake_data()["fake-model"].items():
            seen, unseen = ev.split_classes(task, answers, EXCLUDED)
            self.assertEqual({a["question_id"] for a in unseen},
                             {a["question_id"] for a in answers if old.search(a["question"])}, task)
            self.assertEqual(len(unseen), expected[task], task)
            self.assertEqual(len(seen) + len(unseen), len(answers), task)

    def test_classes_left_out(self):
        rows = by_key(ev.summarize(fake_data(), "test", None, EXCLUDED))
        for task in TASKS:
            seen, unseen = rows[task, "seen_classes"], rows[task, "unseen_classes"]
            self.assertEqual(seen["n"] + unseen["n"], 800, task)
            if task == "id_consistency":  # no COCO class: nothing unseen, so N/A
                self.assertEqual(unseen["n"], 0)
                self.assertTrue(all(unseen[k] is None for k in ev.METRIC_KEYS))
                self.assertEqual(seen["accuracy"], rows[task, "all"]["accuracy"])
            else:
                self.assertGreater(unseen["n"], 0, task)
                self.assertTrue(all(isinstance(unseen[k], float) for k in ev.METRIC_KEYS), task)

    def test_no_class_left_out(self):
        """Empty EXCLUDED_CLASSES: every question is seen, unseen is N/A for every task."""
        rows = by_key(ev.summarize(fake_data(), "test", None, ()))
        for task in TASKS:
            seen, unseen, every = rows[task, "seen_classes"], rows[task, "unseen_classes"], rows[task, "all"]
            self.assertEqual({k: seen[k] for k in ("n", *ev.METRIC_KEYS)},
                             {k: every[k] for k in ("n", *ev.METRIC_KEYS)}, task)
            self.assertEqual(unseen["n"], 0, task)
            self.assertTrue(all(unseen[k] is None for k in (*ev.METRIC_KEYS, "label_yes_ratio")), task)
        self.assertEqual(ev.subset_notes(())["seen_classes"], "every question (no class is left out of training)")

    def test_unknown_class_name_is_an_error(self):
        with self.assertRaises(ValueError):
            ev.summarize(fake_data(), "test", None, ("toasters",))  # a plural, not the COCO name

    def test_na_in_summary_csv_and_figures(self):
        rows = ev.summarize(fake_data(), "test", None, ())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            ev.write_summary(rows, path)
            with open(path) as f:
                written = by_key(list(csv.DictReader(f)))
            for task in TASKS:
                self.assertEqual(written[task, "unseen_classes"]["accuracy"], ev.NA)
                self.assertEqual(written[task, "unseen_classes"]["n"], "0")
                float(written[task, "seen_classes"]["accuracy"])  # a number
            notes = ev.subset_notes(())
            for subset in ("all", "seen_classes", "unseen_classes"):  # unseen is all N/A: drawn without bars
                out = Path(tmp) / f"{subset}.png"
                ev.plot(rows, subset, out, notes[subset] or subset)
                self.assertTrue(out.stat().st_size > 0, subset)


if __name__ == "__main__":
    unittest.main()
