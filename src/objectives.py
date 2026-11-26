"""
    The training methods and their losses. Every method uses the same data, model and number of cases per batch;
    a method sets how batches are made and which terms join the cross-entropy (CE on each case's answer "Yes" /
    "No" and the <|im_end|> after it):
      method        batches                         loss
      sft           single cases (Shuffled SFT)     CE
      group-sft     whole groups (Grouped SFT)      CE
      cf-rank       whole groups                    CE + lambda_f Flip
      cf-invariant  whole groups                    CE + lambda_f Flip + lambda_s Stay
      cf-metric     whole groups                    CE + alpha Align + beta Uniform
    Single cases: the 9,360 cases shuffled one by one, so a batch mixes cases of different groups and its Yes / No
    ratio varies. Whole groups: the 2,340 groups shuffled, A, B, C, D of a group in the same batch (always half
    Yes).
    Terms, per group, averaged over the groups of a batch; m = logit(Yes) - logit(No) at the answer position:
      Flip     1/2 ([gamma - (m_A - m_B)]_+ + [gamma - (m_C - m_D)]_+)       the answer must change across a flip
      Stay     1/2 ([|m_A - m_C| - eps]_+ + [|m_B - m_D| - eps]_+)           and hold across a stay swap
      Align    1/2 (||u_A - u_C||^2 + ||u_B - u_D||^2)                       stay pairs close in feature space
      Uniform  log of the mean of exp(-2 ||u_i - u_j||^2) over the case pairs of the batch and every pair of a
               batch case with a memory-bank feature (many groups)
    u = g(h) / ||g(h)||: h is the last hidden state at the answer position (the one whose logits predict Yes / No),
    g the projection head (Linear H -> H, ReLU, Linear H -> proj_dim), trained with the LoRA and not used at
    inference. The memory bank holds the features of the last bank_size training cases, without gradient: a batch
    of one group (InternVL, --batch-size 4) still spreads against ~64 other groups, and since every model uses the
    same bank, Uniform does not depend on the batch size (the other terms are per-case or per-group means already).
    Bank features come from a few steps back; bank_size 0 uses the batch alone.
"""
import math

import torch
import torch.nn.functional as F

from cf_data import FLIP_PAIRS, STAY_PAIRS

METHODS = {  # name -> (batches of single cases or of whole groups, terms added to CE)
    "sft": ("case", ()),
    "group-sft": ("group", ()),
    "cf-rank": ("group", ("flip",)),
    "cf-invariant": ("group", ("flip", "stay")),
    "cf-metric": ("group", ("align", "uniform")),
}


def margin(logits, yes_id, no_id):
    """m = logit(Yes) - logit(No) at the answer position, one per case."""
    return logits[:, 0, yes_id] - logits[:, 0, no_id]


def sft_loss(logits, batch):
    rows, idx = torch.arange(len(logits)), batch["answer_index"]
    targets = torch.stack([batch["labels"][rows, idx + 1], batch["labels"][rows, idx + 2]], 1).to(logits.device)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


def flip_term(m, gamma):
    """m: margins, groups x cases A, B, C, D."""
    return torch.stack([F.relu(gamma - (m[:, i] - m[:, j])) for i, j in FLIP_PAIRS]).mean()


def stay_term(m, epsilon):
    return torch.stack([F.relu((m[:, i] - m[:, j]).abs() - epsilon) for i, j in STAY_PAIRS]).mean()


def align_term(u):
    """u: unit features, cases in order A, B, C, D per group."""
    g = u.view(-1, 4, u.shape[-1])
    return torch.stack([(g[:, i] - g[:, j]).pow(2).sum(-1) for i, j in STAY_PAIRS]).mean()


def uniform_term(u, bank=None):
    """log of the mean of exp(-2 ||a - b||^2) over the pairs i < j of u and every pair of a row of u with a row of
    bank (no gradient through bank). ||a - b||^2 = 2 - 2 a.b for unit vectors: no sqrt, so no NaN at a = b."""
    i, j = torch.triu_indices(len(u), len(u), 1, device=u.device)
    d2 = 2 - 2 * (u[i] * u[j]).sum(-1)
    if bank is not None and len(bank):
        d2 = torch.cat([d2, (2 - 2 * u @ bank.T).flatten()])
    d2 = d2.clamp_min(0)
    return torch.logsumexp(-2 * d2, 0) - math.log(len(d2))


class Objective(torch.nn.Module):
    """CE plus the method's terms (module docstring). forward(logits, hidden, batch) -> (loss, {term: value});
    its parameters (the projection head, cf-metric only) train with the LoRA. Every training forward pushes its
    features into the memory bank (newest first, at most bank_size). hidden_size: the size of h (the language
    model's hidden size, vlm.lm_head_of(net).in_features); args holds the weights and margins of src/train.py."""

    def __init__(self, terms, collate, hidden_size, args):
        super().__init__()
        self.terms, self.yes_id, self.no_id = terms, collate.yes_id, collate.no_id
        self.weights = {"flip": args.lambda_flip, "stay": args.lambda_stay, "align": args.alpha, "uniform": args.beta}
        self.gamma, self.epsilon, self.bank_size = args.gamma, args.epsilon, args.bank_size
        self.head = None
        if {"align", "uniform"} & set(terms):
            h = hidden_size
            self.head = torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, args.proj_dim))
        self.register_buffer("bank", torch.zeros(0, args.proj_dim), persistent=False)

    def margins(self, logits):
        return margin(logits, self.yes_id, self.no_id).view(-1, 4)  # groups x A, B, C, D

    def features(self, hidden):
        return F.normalize(self.head(hidden.float()), dim=-1)

    def forward(self, logits, hidden, batch):
        parts = {"ce": sft_loss(logits, batch)}
        if self.terms:
            target = batch["target"].view(-1, 4)
            if not (target == target.new_tensor([1, 0, 1, 0])).all():
                raise ValueError("these terms need whole groups, cases in order A, B, C, D")
        if "flip" in self.terms:
            parts["flip"] = flip_term(self.margins(logits), self.gamma)
        if "stay" in self.terms:
            parts["stay"] = stay_term(self.margins(logits), self.epsilon)
        if self.head is not None:
            u = self.features(hidden)
            parts["align"] = align_term(u)
            parts["uniform"] = uniform_term(u, self.bank)
            self.bank = torch.cat([u.detach(), self.bank])[:self.bank_size]
        loss = parts["ce"] + sum(self.weights[k] * parts[k] for k in self.terms)
        return loss, {k: v.item() for k, v in parts.items()}
