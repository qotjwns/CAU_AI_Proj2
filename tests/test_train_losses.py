"""
    The loss terms of src/train.py (Objective) against their formulas, on hand-made logits and hidden states.
    .venv/bin/python -m unittest tests.test_train_losses -v
"""
import math
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import train  # noqa: E402

YES, NO, END, VOCAB, H = 0, 1, 2, 5, 6


def make_batch(margins):
    """Logits whose margin z_Yes - z_No is margins[i] at the answer position, and the batch fields the CE reads
    (answer at index 1, <|im_end|> at index 2, cases in order A, B, C, D per group)."""
    n = len(margins)
    logits = torch.zeros(n, 2, VOCAB)
    logits[:, 0, YES] = torch.tensor(margins, dtype=torch.float) / 2
    logits[:, 0, NO] = -torch.tensor(margins, dtype=torch.float) / 2
    logits[:, 1, END] = 3.0
    target = torch.tensor([1, 0, 1, 0] * (n // 4))
    labels = torch.full((n, 3), -100)
    labels[:, 1] = torch.where(target.bool(), YES, NO)
    labels[:, 2] = END
    return logits, {"answer_index": torch.zeros(n, dtype=torch.long), "labels": labels, "target": target}


def objective(method, **kw):
    args = types.SimpleNamespace(**{"lambda_flip": 1.0, "lambda_stay": 1.0, "gamma": 2.0, "epsilon": 0.5,
                                    "alpha": 1.0, "beta": 1.0, "proj_dim": 4, "bank_size": 256, **kw})
    collate = types.SimpleNamespace(yes_id=YES, no_id=NO)
    net = types.SimpleNamespace(lm_head=torch.nn.Linear(H, VOCAB))  # only its in_features is read
    return train.Objective(train.METHODS[method][1], collate, net, args)


class Terms(unittest.TestCase):
    def test_sft_is_ce_only(self):
        logits, batch = make_batch([3, -3, 3, -3])
        loss, parts = objective("sft")(logits, None, batch)
        self.assertEqual(set(parts), {"ce"})
        torch.testing.assert_close(loss, train.sft_loss(logits, batch))

    def test_flip(self):
        # group 1: m_A - m_B = 1 -> 1, m_C - m_D = 5 -> 0;  group 2: 2 -> 0, -1 -> 3
        logits, batch = make_batch([0.5, -0.5, 2.5, -2.5, 1, -1, -0.5, 0.5])
        loss, parts = objective("cf-rank", lambda_flip=2.0)(logits, None, batch)
        want = ((1 + 0) / 2 + (0 + 3) / 2) / 2
        self.assertAlmostEqual(parts["flip"], want, places=5)
        self.assertAlmostEqual(loss.item(), parts["ce"] + 2.0 * want, places=5)

    def test_stay(self):
        # |m_A - m_C| = 2 -> 1.5, |m_B - m_D| = 0.25 -> 0
        logits, batch = make_batch([3, -2, 1, -2.25])
        _, parts = objective("cf-invariant")(logits, None, batch)
        self.assertAlmostEqual(parts["stay"], (1.5 + 0) / 2, places=5)
        self.assertIn("flip", parts)

    def test_metric(self):
        torch.manual_seed(0)
        obj = objective("cf-metric", proj_dim=3)
        logits, batch = make_batch([1, -1, 1, -1, 1, -1, 1, -1])
        hidden = torch.randn(8, H)
        loss, parts = obj(logits, hidden, batch)
        u = F.normalize(obj.head(hidden), dim=-1)
        align = ((u[0] - u[2]).pow(2).sum() + (u[1] - u[3]).pow(2).sum()
                 + (u[4] - u[6]).pow(2).sum() + (u[5] - u[7]).pow(2).sum()) / 4
        pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
        uniform = math.log(sum(math.exp(-2 * (u[i] - u[j]).pow(2).sum().item()) for i, j in pairs) / len(pairs))
        self.assertAlmostEqual(parts["align"], align.item(), places=5)
        self.assertAlmostEqual(parts["uniform"], uniform, places=5)
        self.assertAlmostEqual(loss.item(), parts["ce"] + parts["align"] + parts["uniform"], places=5)
        loss.backward()  # gradients reach the projection head, and identical features give no NaN
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in obj.parameters()))
        same = obj(logits, torch.zeros(8, H), batch)[0]
        same.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in obj.parameters()))

    def test_memory_bank(self):
        torch.manual_seed(0)
        obj = objective("cf-metric", proj_dim=3, bank_size=6)
        logits, batch = make_batch([1, -1, 1, -1])
        h1, h2 = torch.randn(4, H), torch.randn(4, H)
        obj(logits, h1, batch)
        u1 = F.normalize(obj.head(h1), dim=-1).detach()
        torch.testing.assert_close(obj.bank, u1)  # the first batch, no gradient
        self.assertFalse(obj.bank.requires_grad)
        _, parts = obj(logits, h2, batch)
        u2 = F.normalize(obj.head(h2), dim=-1).detach()
        pairs = [(u2[i], u2[j]) for i in range(4) for j in range(i + 1, 4)] + [(a, b) for a in u2 for b in u1]
        want = math.log(sum(math.exp(-2 * (a - b).pow(2).sum().item()) for a, b in pairs) / len(pairs))
        self.assertAlmostEqual(parts["uniform"], want, places=5)  # 6 batch pairs + 4 x 4 with the bank
        torch.testing.assert_close(obj.bank, torch.cat([u2, u1])[:6])  # newest first, capped at bank_size

    def test_dev_terms(self):
        m = torch.tensor([[3.0, -2.0, 1.0, -2.25]])
        self.assertAlmostEqual(train.flip_term(m, 2.0).item(), (0 + 0) / 2)
        self.assertAlmostEqual(train.stay_term(m, 0.5).item(), (1.5 + 0) / 2)

    def test_group_terms_need_whole_groups(self):
        logits, batch = make_batch([1, -1, 1, -1])
        batch["target"] = torch.tensor([1, 1, 0, 0])  # not A, B, C, D
        with self.assertRaises(ValueError):
            objective("cf-rank")(logits, None, batch)


if __name__ == "__main__":
    unittest.main()
