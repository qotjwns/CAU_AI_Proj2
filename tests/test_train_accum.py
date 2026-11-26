"""
    Gradient accumulation in src/train.py (accumulation_plan, planned_steps, fit), on a tiny linear model, and the
    planned step counts of both methods on the real COCO existence splits.
    .venv/bin/python -m unittest tests.test_train_accum -v
"""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import train  # noqa: E402
from cf_data import SPLITS, CaseDataset, GroupDataset  # noqa: E402


def run_fit(n_batches, grad_accum, epochs, max_steps=None):
    """fit() with learning rate 0, so the parameters never change and every step's gradient can be checked
    against the mean of its batches' gradients computed one by one."""
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    params = list(model.parameters())
    stepped = []  # the gradient each optimizer step saw

    class RecordingSGD(torch.optim.SGD):
        def step(self, closure=None):
            stepped.append([p.grad.clone() for p in params])
            return super().step(closure)

    opt = RecordingSGD(params, lr=0.0)
    steps = train.planned_steps(n_batches, grad_accum, epochs, max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    g = torch.Generator().manual_seed(1)
    loader = [(0.1 * torch.randn(2, 3, generator=g), 0.1 * torch.randn(2, 1, generator=g)) for _ in range(n_batches)]
    loss_of = lambda b: torch.nn.functional.mse_loss(model(b[0]), b[1])
    per_batch = [torch.autograd.grad(loss_of(b), params) for b in loader]
    ends = []
    taken = train.fit(params, opt, sched, loader, loss_of, epochs, steps, grad_accum,
                      on_epoch=lambda step, epoch: ends.append((epoch, step)))
    return {"taken": taken, "steps": steps, "sched": sched.last_epoch, "stepped": stepped,
            "per_batch": per_batch, "epoch_ends": ends, "grads_left": [p.grad for p in params]}


def expected_grads(per_batch, grad_accum, epochs, steps):
    """Mean gradient of each step's batches: groups of grad_accum per epoch, the last one shorter."""
    out = []
    for _ in range(epochs):
        for start in range(0, len(per_batch), grad_accum):
            group = per_batch[start:start + grad_accum]
            out.append([sum(g[i] for g in group) / len(group) for i in range(len(group[0]))])
    return out[:steps]


class Plan(unittest.TestCase):
    def test_divisible(self):
        plan = train.accumulation_plan(6, 2)
        self.assertEqual([s for s, _ in plan], [0.5] * 6)
        self.assertEqual([e for _, e in plan], [False, True] * 3)

    def test_remainder(self):
        plan = train.accumulation_plan(7, 3)
        self.assertEqual([s for s, _ in plan], [1 / 3] * 6 + [1.0])  # the leftover batch alone: scale 1
        self.assertEqual([i for i, (_, e) in enumerate(plan) if e], [2, 5, 6])

    def test_accumulation_longer_than_epoch(self):
        plan = train.accumulation_plan(3, 5)
        self.assertEqual(plan, [(1 / 3, False), (1 / 3, False), (1 / 3, True)])

    def test_no_accumulation(self):
        self.assertEqual(train.accumulation_plan(4, 1), [(1.0, True)] * 4)

    def test_planned_steps(self):
        self.assertEqual(train.planned_steps(6, 2, 2), 6)
        self.assertEqual(train.planned_steps(7, 3, 2), 6)  # ceil(7 / 3) = 3 per epoch
        self.assertEqual(train.planned_steps(7, 2, 3, max_steps=5), 5)
        self.assertEqual(train.planned_steps(1170, 1, 1), 1170)


class Fit(unittest.TestCase):
    def check(self, n_batches, grad_accum, epochs, max_steps=None):
        r = run_fit(n_batches, grad_accum, epochs, max_steps)
        self.assertEqual(r["taken"], r["steps"])  # optimizer steps = planned
        self.assertEqual(r["sched"], r["steps"])  # scheduler steps = planned
        want = expected_grads(r["per_batch"], grad_accum, epochs, r["steps"])
        self.assertEqual(len(r["stepped"]), len(want))
        for i, (got, exp) in enumerate(zip(r["stepped"], want)):
            self.assertLess(torch.cat([g.flatten() for g in got]).norm().item(), 1.0)  # below the clip
            for a, b in zip(got, exp):
                torch.testing.assert_close(a, b, msg=f"step {i + 1}")
        self.assertTrue(all(g is None for g in r["grads_left"]))  # nothing left after the run
        return r

    def test_divisible(self):
        r = self.check(6, 2, epochs=2)
        self.assertEqual(r["epoch_ends"], [(1, 3), (2, 6)])

    def test_remainder_is_scaled_and_does_not_cross_epochs(self):
        r = self.check(7, 3, epochs=2)
        self.assertEqual(r["epoch_ends"], [(1, 3), (2, 6)])
        torch.testing.assert_close(r["stepped"][2][0], r["per_batch"][6][0])  # leftover batch at full weight
        first = [(a + b + c) / 3 for a, b, c in zip(*r["per_batch"][:3])]
        torch.testing.assert_close(r["stepped"][3][0], first[0])  # epoch 2 starts clean

    def test_accumulation_longer_than_epoch(self):
        self.check(3, 5, epochs=2)

    def test_no_accumulation(self):
        self.check(5, 1, epochs=1)

    def test_max_steps_mid_epoch(self):
        r = self.check(7, 3, epochs=3, max_steps=4)
        self.assertEqual(r["epoch_ends"], [(1, 3), (2, 4)])  # the cut epoch still ends (adapter, dev)


@unittest.skipUnless((SPLITS / "train.jsonl").exists(), "data/coco_existence not built")
class RealSplits(unittest.TestCase):
    """Both methods plan the same optimizer steps, as main() builds their loaders."""

    def steps(self, split, method, batch_size, grad_accum):
        batching = train.METHODS[method][0]
        data = CaseDataset(split) if batching == "case" else GroupDataset(split)
        loader = DataLoader(data, batch_size=train.items_per_batch(batching, batch_size))
        return train.planned_steps(len(loader), grad_accum, epochs=1)

    def test_methods_agree(self):
        for split, at_8 in (("pilot_train", 234), ("train", 1170)):
            for batch_size, grad_accum, want in ((8, 1, at_8), (4, 2, at_8), (12, 7, None)):
                got = {m: self.steps(split, m, batch_size, grad_accum) for m in train.METHODS}
                self.assertEqual(len(set(got.values())), 1, (split, batch_size, grad_accum, got))
                if want:
                    self.assertEqual(got["sft"], want, (split, batch_size, grad_accum))


class Tiny(torch.nn.Module):
    """Stands in for the VLM (and its peft wrapper) in main()."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 2)

    def forward(self, x):
        return self.lin(x)

    def gradient_checkpointing_enable(self, **kw):
        pass

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def get_base_model(self):
        return self


class TinyCollate:
    template, yes_id, no_id = "{question}", 0, 1

    def __call__(self, items):
        return {"x": torch.ones(sum(len(x) if isinstance(x, list) else 1 for x in items), 1)}


@unittest.skipUnless((SPLITS / "pilot_train.jsonl").exists(), "data/coco_existence not built")
class Main(unittest.TestCase):
    """main() end to end on the real pilot split, with a tiny model in place of the VLM: loaders, callbacks,
    logging and the step check run as in a real run (no VLM is loaded)."""

    def run_main(self, method, batch_size, grad_accum, epochs, tmp):
        fake_peft = types.SimpleNamespace(LoraConfig=lambda **kw: None, get_peft_model=lambda m, c: m,
                                          __version__="fake")
        argv = ["train.py", "--model", "qwen2.5-vl-3b", "--method", method, "--train", "pilot_train",
                "--batch-size", str(batch_size), "--grad-accum", str(grad_accum), "--epochs", str(epochs),
                "--workers", "0", "--log-every", "1", "--tag", "test"]
        with mock.patch.multiple(train, PROJECT=Path(tmp), pick_device=lambda: "cpu",
                                 load_model=lambda name, model_dir: (Tiny(), Tiny(), TinyCollate()),
                                 answer_logits=lambda model, batch, device: model(batch["x"]),
                                 evaluate=lambda *a: {"acc": 0.0},
                                 METHODS={m: (b, lambda logits, batch: logits.pow(2).mean())
                                          for m, (b, _) in train.METHODS.items()}), \
                mock.patch.dict(sys.modules, {"peft": fake_peft}), mock.patch.object(sys, "argv", argv), \
                mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            train.main()
        run = Path(tmp) / "runs" / method / "qwen2.5-vl-3b-test"
        args = json.loads((run / "args.json").read_text())
        log = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
        return args, log

    def test_both_methods_take_the_planned_steps(self):
        for method in train.METHODS:
            for batch_size, grad_accum, epochs, per_epoch in ((8, 1, 1, 234), (4, 5, 2, 94)):  # 468 / 5: rest 3
                with tempfile.TemporaryDirectory() as tmp:
                    args, log = self.run_main(method, batch_size, grad_accum, epochs, tmp)
                steps = [r["step"] for r in log if "loss" in r]
                dev = [(r["epoch"], r["step"]) for r in log if "dev" in r]
                case = (method, batch_size, grad_accum)
                self.assertEqual(args["steps"], per_epoch * epochs, case)
                self.assertEqual(steps, list(range(1, per_epoch * epochs + 1)), case)
                self.assertEqual(dev, [(0, 0)] + [(e, e * per_epoch) for e in range(1, epochs + 1)], case)


if __name__ == "__main__":
    unittest.main()
