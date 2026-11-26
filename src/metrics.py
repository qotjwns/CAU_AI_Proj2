"""
    Metrics from the answer margin m = logit(Yes) - logit(No), for groups of cases A, B, C, D (Yes, No, Yes, No):
      acc        m > 0 exactly on the Yes cases          yes_ratio  share of cases with m > 0
      group_acc  all 4 cases of a group right            flip_rank  m_A > m_B and m_C > m_D, per flip pair
      stay_gap   mean |m_A - m_C| and |m_B - m_D|: 0 when the answer ignores the swapped P image
    Pure functions of the margins, so any model run that records margins can use them.
"""
import torch

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
