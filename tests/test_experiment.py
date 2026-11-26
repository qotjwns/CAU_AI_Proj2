"""
    src/experiment.py runs train -> MIHBench inference with the adapter -> scoring, in order, with the right
    arguments, and stops at a failing step. No step really runs (subprocess.call is replaced).
    .venv/bin/python -m unittest tests.test_experiment -v
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import experiment  # noqa: E402


def steps(argv, fail_at=None):
    """[(script, args)] experiment.main() would run; fail_at: the script that returns exit code 1."""
    calls = []

    def fake_call(cmd, cwd):
        script, args = Path(cmd[1]).name, cmd[2:]
        calls.append((script, args))
        return 1 if script == fail_at else 0

    with mock.patch.object(sys, "argv", ["experiment.py", *argv]), \
            mock.patch.object(experiment.subprocess, "call", fake_call), mock.patch("sys.stdout"):
        try:
            experiment.main()
        except SystemExit as e:
            calls.append(("exit", str(e.code)))
    return calls


class Chain(unittest.TestCase):
    def test_order_and_arguments(self):
        calls = steps(["--model", "qwen2.5-vl-3b", "--method", "cf-invariant", "--train", "pilot_train",
                       "--tag", "pilot", "--lambda-stay", "0.5"])
        self.assertEqual(calls, [
            ("train.py", ["--model", "qwen2.5-vl-3b", "--method", "cf-invariant", "--tag", "pilot",
                          "--train", "pilot_train", "--lambda-stay", "0.5"]),
            ("infer_mihbench.py", ["--adapter", str(Path("runs/cf-invariant/qwen2.5-vl-3b-pilot")), "--task", "all"]),
            ("eval_mihbench.py", ["--method", "cf-invariant"]),
        ])

    def test_vanilla_and_limits(self):
        calls = steps(["--model", "qwen2.5-vl-3b", "--method", "sft", "--mihbench-tasks", "existence_random",
                       "--mihbench-limit", "5", "--vanilla"])
        self.assertEqual([c[0] for c in calls], ["train.py", "infer_mihbench.py", "eval_mihbench.py",
                                                  "infer_mihbench.py", "eval_mihbench.py"])
        self.assertEqual(calls[1][1], ["--adapter", str(Path("runs/sft/qwen2.5-vl-3b")),
                                       "--task", "existence_random", "--limit", "5"])
        self.assertEqual(calls[3][1], ["--model", "qwen2.5-vl-3b", "--task", "existence_random", "--limit", "5"])
        self.assertEqual(calls[4][1], ["--method", "vanilla"])

    def test_stops_at_failure(self):
        calls = steps(["--model", "qwen2.5-vl-3b", "--method", "sft"], fail_at="train.py")
        self.assertEqual([c[0] for c in calls], ["train.py", "exit"])

    def test_skip_train(self):
        calls = steps(["--model", "qwen2.5-vl-3b", "--method", "sft", "--tag", "pilot", "--skip-train"])
        self.assertEqual([c[0] for c in calls], ["infer_mihbench.py", "eval_mihbench.py"])
        self.assertEqual([c[0] for c in steps(["--model", "m", "--method", "sft", "--skip-train", "--epochs", "2"])],
                         ["exit"])


if __name__ == "__main__":
    unittest.main()
