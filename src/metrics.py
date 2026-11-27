"""
    Metrics from the answer margin m = logit(Yes) - logit(No), for groups of cases A, B, C, D (Yes, No, Yes, No):
      acc        m > 0 exactly on the Yes cases          yes_ratio  share of cases with m > 0
      group_acc  all 4 cases of a group right            flip_rank  m_A > m_B and m_C > m_D, per flip pair
      stay_gap   mean |m_A - m_C| and |m_B - m_D|: 0 when the answer ignores the swapped P image
    Pure functions of the margins, so any model run that records margins can use them.
    And the geometry of a set of features (one row per case), to tell collapse from spread:
      cos_mean   mean cosine over all pairs of rows: 1 when every row points the same way, about 0 for random
                 directions
      erank      effective rank of the rows around their mean (exp of the entropy of the normalized singular
                 values, Roy & Vetterli 2007): how many directions the spread uses, 0 when every row is the same
    Rows can be tightly bunched (cos_mean near 1) and still linearly separable: a probe follows the small spread.
"""
import torch
import torch.nn.functional as F

from cf_data import ANSWERS, FLIP_PAIRS, STAY_PAIRS


def margin_metrics(m):
    """m: margins, groups x cases A, B, C, D."""
    right = (m > 0) == torch.tensor([a == "Yes" for a in ANSWERS])
    return {"groups": len(m),
            "acc": round(right.float().mean().item(), 4),
            "yes_ratio": round((m > 0).float().mean().item(), 4),
            "group_acc": round(right.all(1).float().mean().item(), 4),
            "flip_rank": round(torch.stack([m[:, i] > m[:, j] for i, j in FLIP_PAIRS]).float().mean().item(), 4),
            "stay_gap": round(torch.stack([(m[:, i] - m[:, j]).abs() for i, j in STAY_PAIRS]).mean().item(), 3)}


def feature_geometry(x):
    """x: features, one row per case (at least 2). -> {cos_mean, erank} (module docstring)."""
    x = x.double()
    u = F.normalize(x, dim=-1)
    s = u.sum(0)
    cos_mean = (s @ s - (u * u).sum()) / (len(x) * (len(x) - 1))  # sum over i != j of u_i . u_j
    sv = torch.linalg.svdvals(x - x.mean(0))
    sv = sv[sv > sv.max() * 1e-9] if sv.max() > 0 else sv[:0]
    p = sv / sv.sum()
    erank = torch.exp(-(p * p.log()).sum()).item() if len(p) else 0.0
    return {"cos_mean": round(cos_mean.item(), 4), "erank": round(erank, 2)}
