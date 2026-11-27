"""
    The training diagnostics without any model weights: src/metrics.py feature_geometry and src/runner.py
    term_gradients, on tiny modules (with dropout and non-reentrant gradient checkpointing, as in training).
    .venv/bin/python -m unittest tests.test_diagnostics -v
"""
import math
import sys
import unittest
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import feature_geometry  # noqa: E402
from runner import term_gradients  # noqa: E402


class Geometry(unittest.TestCase):
    def test_one_point(self):
        g = feature_geometry(torch.tensor([[1.0, 2.0, 3.0]]).repeat(5, 1))
        self.assertEqual(g, {"cos_mean": 1.0, "erank": 0.0})

    def test_orthogonal_rows(self):
        g = feature_geometry(torch.eye(4))
        self.assertAlmostEqual(g["cos_mean"], 0.0, places=6)
        self.assertAlmostEqual(g["erank"], 3.0, places=2)  # 4 rows around their mean span 3 directions

    def test_bunched_but_spread_in_two_directions(self):
        """Far from the origin along e3, with a small spread along e1 and e2: nearly one direction (cos_mean near
        1), yet the spread uses 2 directions, which a linear probe can read."""
        x = torch.tensor([[1.0, 0, 100, 0], [-1, 0, 100, 0], [0, 1, 100, 0], [0, -1, 100, 0]])
        g = feature_geometry(x)
        self.assertGreater(g["cos_mean"], 0.999)
        self.assertAlmostEqual(g["erank"], 2.0, places=2)

    def test_cos_mean_is_the_mean_over_pairs(self):
        torch.manual_seed(0)
        x = torch.randn(6, 5)
        u = torch.nn.functional.normalize(x, dim=-1)
        want = sum((u[i] @ u[j]).item() for i in range(6) for j in range(6) if i != j) / 30
        self.assertAlmostEqual(feature_geometry(x)["cos_mean"], round(want, 4), places=4)
        self.assertEqual(feature_geometry(3 * x)["cos_mean"], feature_geometry(x)["cos_mean"])  # scale-free


class Net(torch.nn.Module):
    """A 'LoRA' part (with dropout, checkpointed) and a 'head' part on top of it."""

    def __init__(self):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.5), torch.nn.Tanh())
        self.head = torch.nn.Linear(4, 2)

    def forward(self, x):
        h = checkpoint(self.body, x, use_reentrant=False)
        return h, self.head(h)


class TermGradients(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.net = Net().train()
        self.groups = {"lora": list(self.net.body.parameters()), "head": list(self.net.head.parameters())}
        h, z = self.net(torch.randn(5, 3))
        self.parts = {"ce": h.pow(2).mean(), "align": z.pow(2).mean(), "uniform": (z.sum() * h.sum()).sin()}
        self.weights = {"align": 0.5, "uniform": 2.0}

    def manual(self, term):
        w = self.weights.get(term, 1.0)
        params = [p for ps in self.groups.values() for p in ps]
        return torch.autograd.grad(w * self.parts[term], params, retain_graph=True, allow_unused=True)

    def test_against_autograd(self):
        got = term_gradients(self.parts, self.weights, self.groups)
        n_lora = len(self.groups["lora"])
        flat = {}
        for term in self.parts:
            grads = [torch.zeros_like(p) if g is None else g
                     for p, g in zip([p for ps in self.groups.values() for p in ps], self.manual(term))]
            lora, head = torch.cat([g.flatten() for g in grads[:n_lora]]), torch.cat([g.flatten() for g in grads[n_lora:]])
            flat[term] = lora
            self.assertAlmostEqual(got["norm"]["lora"][term], lora.norm().item(), places=5)
            self.assertAlmostEqual(got["norm"]["head"][term], head.norm().item(), places=5)
        self.assertEqual(got["norm"]["head"]["ce"], 0.0)  # CE does not reach the head
        for term in ("align", "uniform"):
            want = torch.nn.functional.cosine_similarity(flat[term], flat["ce"], dim=0).item()
            self.assertAlmostEqual(got["cos_ce"][term], want, places=4)
        self.assertEqual(set(got["cos_ce"]), {"align", "uniform"})

    def test_leaves_grad_and_graph_for_the_step(self):
        """.grad untouched, the graph kept, and the terms' gradients add up to the step's own gradient (the
        checkpointed dropout is recomputed with the same mask every time)."""
        got = term_gradients(self.parts, self.weights, self.groups)
        self.assertTrue(all(p.grad is None for p in self.net.parameters()))
        per_term = {t: self.manual(t) for t in self.parts}
        loss = self.parts["ce"] + sum(self.weights[t] * self.parts[t] for t in ("align", "uniform"))
        loss.backward()
        params = [p for ps in self.groups.values() for p in ps]
        for i, p in enumerate(params):
            total = sum(g[i] for g in per_term.values() if g[i] is not None)
            torch.testing.assert_close(p.grad, total)
        lora_norm = math.sqrt(sum(p.grad.pow(2).sum().item() for p in self.groups["lora"]))
        self.assertGreater(lora_norm, 0)
        self.assertIn("ce", got["norm"]["lora"])

    def test_zero_gradient_and_empty_group(self):
        parts = {"ce": self.parts["ce"], "align": self.parts["align"]}
        got = term_gradients(parts, {"align": 0.0}, {"lora": self.groups["lora"], "head": []})
        self.assertEqual(set(got["norm"]), {"lora"})
        self.assertEqual(got["norm"]["lora"]["align"], 0.0)
        self.assertIsNone(got["cos_ce"]["align"])


if __name__ == "__main__":
    unittest.main()
