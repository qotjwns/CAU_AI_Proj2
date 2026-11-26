"""
    The training and dev loops every method shares (src/train.py builds what they run on).
      fit       optimizer loop with gradient accumulation: --grad-accum batches per optimizer step. When an epoch's
                batch count does not divide evenly, its last step takes the batches left over, each loss scaled by
                1 / their number, so an epoch always ends with a step and no gradient carries into the next one:
                ceil(batches / grad_accum) steps per epoch (planned_steps). Gradients are cleared at every epoch
                start and after every step.
      evaluate  dev metrics (src/metrics.py) and the loss terms on dev, with the run's gamma and epsilon, for every
                method (so an SFT run shows them too): ce, flip, stay; cf-metric adds align and uniform (over all
                dev case pairs, no memory bank)
    Neither loop knows the model: fit gets loss_of(batch), evaluate gets forward(batch) -> (logits, hidden)
    (models.<Family>.answer_outputs in src/train.py).
"""
import math

import torch
from tqdm import tqdm

from metrics import margin_metrics
from objectives import align_term, flip_term, margin, sft_loss, stay_term, uniform_term


def items_per_batch(batching, batch_size):
    """DataLoader batch size: batch_size cases, as single cases or as whole groups of 4."""
    return batch_size // (1 if batching == "case" else 4)


def accumulation_plan(n_batches, grad_accum):
    """[(loss scale, optimizer step after this batch)] for one epoch of n_batches. Batches go in groups of
    grad_accum; a shorter last group is scaled by 1 / its own size and still ends with a step."""
    plan = []
    for start in range(0, n_batches, grad_accum):
        size = min(grad_accum, n_batches - start)
        plan += [(1 / size, j == size - 1) for j in range(size)]
    return plan


def planned_steps(n_batches, grad_accum, epochs, max_steps=None):
    """Optimizer steps the run will take: ceil(n_batches / grad_accum) per epoch, capped at max_steps."""
    steps = epochs * math.ceil(n_batches / grad_accum)
    return min(steps, max_steps) if max_steps else steps


def fit(params, opt, sched, loader, loss_of, epochs, steps, grad_accum, on_step=None, on_epoch=None):
    """The optimizer loop every method shares. loss_of(batch) -> loss; on_step(step, epoch, mean loss of the
    step's batches, grad norm) after each optimizer step; on_epoch(step, epoch) after each epoch (also the last,
    cut short by steps). Gradients are cleared at every epoch start and after every step, so none crosses epochs.
    Returns the optimizer steps taken."""
    step = 0
    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        losses = []
        for (scale, ends_step), batch in zip(accumulation_plan(len(loader), grad_accum), loader):
            loss = loss_of(batch)
            (loss * scale).backward()
            losses.append(loss.item())
            if not ends_step:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if on_step:
                on_step(step, epoch, sum(losses) / len(losses), grad_norm)
            losses = []
            if step >= steps:
                break
        if on_epoch:
            on_epoch(step, epoch)
        if step >= steps:
            break
    return step


@torch.no_grad()
def evaluate(loader, collate, objective, forward):
    """Dev metrics and the loss terms on dev (module docstring); the memory bank is neither read nor filled.
    The caller puts the model in eval mode first and back in training mode after."""
    margins, losses, feats = [], [], []
    for batch in tqdm(loader, desc="dev", leave=False):
        logits, hidden = forward(batch)
        margins.append(margin(logits, collate.yes_id, collate.no_id).cpu())
        losses.append(sft_loss(logits, batch).item())
        if objective.head is not None:
            feats.append(objective.features(hidden).cpu())
    m = torch.cat(margins).view(-1, 4)  # groups x cases A, B, C, D
    terms = {"ce": sum(losses) / len(losses), "flip": flip_term(m, objective.gamma).item(),
             "stay": stay_term(m, objective.epsilon).item()}
    if feats:
        u = torch.cat(feats)
        terms.update(align=align_term(u).item(), uniform=uniform_term(u).item())
    return {**margin_metrics(m), "terms": {k: round(v, 4) for k, v in terms.items()}}
