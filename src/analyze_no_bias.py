#!/usr/bin/env python3
"""
    Why does a model answer "No" to MIHBench existence questions whose label is yes? Reads the saved answers
    (results/mihbench/<method>/<model>/existence_*.jsonl) and the COCO val2014 annotations; no model is run.
      figures/mihbench/<method>/no_bias.png    four panels:
        a  says yes when the object is annotated in every image, by its area where it is smallest
        b  says yes on yes questions the COCO annotations agree with vs. those they contradict
        c  when the object is in every image but the answer is No, which image the answer says lacks it
        d  when exactly one image lacks the object, how often the answer names that image (chance 1/3)
      results/mihbench/<method>/no_bias.json   the numbers behind the panels
    Panels c and d read the explanation after "No", so they only cover models that explain (Qwen); the regex
    reading is a heuristic, checked by hand on samples.
    python src/analyze_no_bias.py                                   # Qwen 7B vs 3B, method vanilla
    python src/analyze_no_bias.py --focus qwen2.5-vl-7b --compare qwen2.5-vl-3b
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from coco_pools import COCO, MIH_COCO, MIH_QUESTION, load_coco
from eval_mihbench import AXIS, GRID, INK, INK_2, SURFACE, assign_colors, display_name, load, rounded_bar

PROJECT = Path(__file__).resolve().parents[1]
QUESTIONS = PROJECT / "data" / "mihbench" / "questions"
TASKS = ["existence_random", "existence_popular", "existence_adversarial"]
SIZE_BINS = [(0, 0.005, "<0.5%"), (0.005, 0.02, "0.5–2%"), (0.02, 0.08, "2–8%"), (0.08, 1.01, "≥8%")]
ORDINAL = {"first": 1, "second": 2, "third": 3, "1": 1, "2": 2, "3": 3}
NEGATION = re.compile(r"\b(no|not|without|absent|lacks?|none|neither|nor)\b|n't", re.I)


def noun_pattern(obj):
    """The asked class as it shows up in answers: its last word, singular or plural."""
    head = obj.split()[-1]
    forms = {head, head + "s", head + "es", head[:-1] if head.endswith("s") else head}
    if head == "person":
        forms |= {"people", "persons"}
    return re.compile(r"\b(" + "|".join(map(re.escape, forms)) + r")\b", re.I)


def positions(sentence, n):
    hits = {ORDINAL[w.lower()] for w in re.findall(r"\b(first|second|third)\b(?=[^.]*?\bimages?\b)", sentence, re.I)}
    hits |= {ORDINAL[w] for w in re.findall(r"\bimage (\d)\b", sentence, re.I)}
    if re.search(r"first two images", sentence, re.I):
        hits |= {1, 2}
    if re.search(r"last two images", sentence, re.I):
        hits |= {n - 1, n}
    if re.search(r"\blast image", sentence, re.I):
        hits.add(n)
    return hits


def sentences(text):
    return re.split(r"(?<=[.!?])\s+", text)


def claimed_missing(text, obj, n=3):
    """Which images the answer says lack the object: ('images', {i, ...}), ('none', set()) for 'in none / any of
    the images', ('not_all', set()) for only 'not in all images', or ('unspecified', set())."""
    noun, found, none, not_all = noun_pattern(obj), set(), False, False
    for s in sentences(text):
        if not NEGATION.search(s) or not noun.search(s):
            continue
        hits = positions(s, n)
        if hits:
            found |= hits
        elif re.search(r"\b(any|none) of the\b|\bneither\b", s, re.I):
            none = True
        elif re.search(r"\ball (three |3 |the )?images\b|\bnot all\b", s, re.I):
            not_all = True
    if found:
        return ("images", found) if len(found) < n else ("none", found)
    return ("none", set()) if none else ("not_all", set()) if not_all else ("unspecified", set())


def affirmed(text, obj, n=3):
    """Images the explanation (after the opening verdict) describes as holding the object, without negation."""
    noun, found = noun_pattern(obj), set()
    for s in sentences(text)[1:]:
        if noun.search(s) and not NEGATION.search(s):
            found |= positions(s, n)
    return found


def coco_presence():
    """{(task, question_id): (object, [largest area ratio of the object per image, None if absent])}"""
    _, classes, present = load_coco(COCO / "annotations" / "instances_val2014.json")
    by_name = {c["name"]: i for i, c in classes.items()}
    out = {}
    for task in TASKS:
        for line in open(QUESTIONS / f"{task}.jsonl"):
            q = json.loads(line)
            obj = MIH_QUESTION.fullmatch(q["text"]).group(1)
            out[task, q["question_id"]] = (obj, [present.get(int(MIH_COCO.fullmatch(name).group(1)), {})
                                                 .get(by_name[obj]) for name in q["image_list"]])
    return out


def analyze(data, coco):
    """Per model: yes rates by COCO agreement and object size, and for models that explain, what the No says."""
    stats = {}
    for model, tasks in data.items():
        answers = {(t, a["question_id"]): a for t in TASKS for a in tasks.get(t, [])}
        if len(answers) != 800 * len(TASKS):
            continue
        yes_q = [k for k, a in answers.items() if a["label"] == "yes"]
        agree = [k for k in yes_q if all(x is not None for x in coco[k][1])]
        disagree = [k for k in yes_q if any(x is None for x in coco[k][1])]
        rate = lambda ks: sum(answers[k]["pred"] == "yes" for k in ks) / len(ks)
        s = {"recall_coco_agrees": rate(agree), "n_coco_agrees": len(agree),
             "says_yes_coco_disagrees": rate(disagree), "n_coco_disagrees": len(disagree), "recall_by_size": []}
        for lo, hi, label in SIZE_BINS:
            ks = [k for k in agree if lo <= min(coco[k][1]) < hi]
            s["recall_by_size"].append({"bin": label, "n": len(ks), "recall": rate(ks)})
        explains = sum(len(a["text"].split()) for a in answers.values()) / len(answers) > 5
        if explains:
            wrong = [k for k in agree if answers[k]["pred"] == "no"]
            kinds, where = Counter(), Counter()
            for k in wrong:
                kind, imgs = claimed_missing(answers[k]["text"], coco[k][0])
                kinds[f"{len(imgs)} image" if kind == "images" else kind] += 1
                if kind == "images" and len(imgs) == 1:
                    where[next(iter(imgs))] += 1
            contradicts = sum(affirmed(answers[k]["text"], coco[k][0]) == {1, 2, 3} for k in wrong)
            one_missing = [k for k in answers if sum(x is None for x in coco[k][1]) == 1]
            named = right = 0
            for k in one_missing:
                if answers[k]["pred"] != "no":
                    continue
                kind, imgs = claimed_missing(answers[k]["text"], coco[k][0])
                if kind == "images" and len(imgs) == 1:
                    named += 1
                    right += next(iter(imgs)) - 1 == [x is None for x in coco[k][1]].index(True)
            s.update({"wrong_no_when_in_every_image": len(wrong), "what_the_no_claims": dict(kinds),
                      "single_image_claim_position": {str(p): where[p] for p in (1, 2, 3)},
                      "explanation_affirms_all_three": contradicts,
                      "one_image_truly_missing": {"questions": len(one_missing), "names_one_image": named,
                                                  "names_the_right_one": right}})
        stats[model] = s
    return stats


def style(ax, title, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=9.5, fontweight="bold", pad=8)
    ax.tick_params(colors=INK_2, length=0, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=8)


def bars(ax, groups, series, values, colors, group=0.72, gap=0.02):
    """Grouped rounded bars: values[(series, group)] in percent."""
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.figure.canvas.draw()
    width = group / len(series)
    for i, g in enumerate(groups):
        for j, s in enumerate(series):
            if (s, g) in values:
                rounded_bar(ax, i - group / 2 + j * width + gap / 2, width - gap, values[s, g], colors[s])


def plot(stats, focus, compare, out_path):
    models = list(stats)
    colors = assign_colors(models)
    explained = [m for m in (compare, focus) if m in stats and "what_the_no_claims" in stats[m]]
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]})
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), dpi=200, facecolor=SURFACE)
    (a, b), (c, d) = axes

    style(a, "a  Says yes when the object is in every image", "% of yes questions")
    x = range(len(SIZE_BINS))
    for m in models:
        ys = [100 * r["recall"] for r in stats[m]["recall_by_size"]]
        a.plot(x, ys, color=colors[m], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
               markeredgewidth=1.5, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ns = [r["n"] for r in stats[models[0]]["recall_by_size"]]
    a.set_xticks(list(x), [f"{lab}\nn={n}" for (_, _, lab), n in zip(SIZE_BINS, ns)])
    a.set_xlabel("object area where it is smallest (COCO)", color=INK_2, fontsize=8)
    a.set_xlim(-0.3, len(SIZE_BINS) - 0.7)
    a.set_ylim(0, 100)

    style(b, "b  Says yes on yes questions, by COCO", "% answered yes")
    b.set_ylim(0, 100)
    n_agree, n_dis = stats[models[0]]["n_coco_agrees"], stats[models[0]]["n_coco_disagrees"]
    groups = [f"COCO: in every image\nn={n_agree}", f"COCO: missing in ≥1 image\nn={n_dis}"]
    b.set_xticks(range(2), groups)
    bars(b, groups, models, {**{(m, groups[0]): 100 * stats[m]["recall_coco_agrees"] for m in models},
                             **{(m, groups[1]): 100 * stats[m]["says_yes_coco_disagrees"] for m in models}}, colors)

    style(c, "c  Image the No says lacks it (object in all three)", "% of one-image claims")
    c.set_ylim(0, 100)
    pos = ["image 1", "image 2", "image 3"]
    c.set_xticks(range(3), pos)
    shares = {}
    for m in explained:
        where = stats[m]["single_image_claim_position"]
        total = sum(where.values())
        shares.update({(m, p): 100 * where[str(i + 1)] / total for i, p in enumerate(pos)})
    bars(c, pos, explained, shares, colors)
    c.axhline(100 / 3, color=INK_2, linewidth=0.8)
    c.text(0.5, 100 / 3 + 1.5, "even", color=INK_2, fontsize=7.5, ha="center", va="bottom")  # in the gap
    c.text(-0.45, 96, "  ".join(f"{display_name(m)}: n={sum(stats[m]['single_image_claim_position'].values())}"
                               for m in explained), color=INK_2, fontsize=7, va="top")

    style(d, "d  Names the image that truly lacks it", "% of one-image claims")
    d.set_ylim(0, 100)
    names = [display_name(m) for m in explained]
    d.set_xticks(range(len(explained)), names)
    d.set_xlim(-0.5, len(explained) - 0.5)
    fig.canvas.draw()
    for i, m in enumerate(explained):  # one bar per model, centered on its tick
        t = stats[m]["one_image_truly_missing"]
        v = 100 * t["names_the_right_one"] / t["names_one_image"]
        rounded_bar(d, i - 0.18, 0.36, v, colors[m])
        d.text(i, v + 2, f"{v:.0f}%  (n={t['names_one_image']})", color=INK, fontsize=8, ha="center", va="bottom")
    d.axhline(100 / 3, color=INK_2, linewidth=0.8)
    d.text(0.5, 100 / 3 + 1.5, "chance", color=INK_2, fontsize=7.5, ha="center", va="bottom")  # in the gap

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[m], linewidth=0) for m in models]
    fig.legend(handles, [display_name(m) for m in models], loc="upper center", ncol=len(models), frameon=False,
               fontsize=8.5, labelcolor=INK_2, handlelength=0.9, handleheight=0.9, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, "MIHBench existence (random, popular, adversarial; 2,400 questions). Panels c and d read "
             "the explanation after the verdict, so only models that explain are shown.",
             ha="center", va="bottom", color=INK_2, fontsize=7)
    fig.tight_layout(rect=(0, 0.025, 1, 0.95), h_pad=2.5, w_pad=2.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="vanilla")
    ap.add_argument("--focus", default="qwen2.5-vl-7b")
    ap.add_argument("--compare", default="qwen2.5-vl-3b")
    ap.add_argument("--results-dir", type=Path, default=PROJECT / "results" / "mihbench")
    ap.add_argument("--figures-dir", type=Path, default=PROJECT / "figures" / "mihbench")
    args = ap.parse_args()
    method_dir = args.results_dir / args.method
    stats = analyze(load(method_dir), coco_presence())
    if args.focus not in stats:
        raise SystemExit(f"{args.focus}: no complete existence answers under {method_dir}")
    (method_dir / "no_bias.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    plot(stats, args.focus, args.compare, args.figures_dir / args.method / "no_bias.png")
    for m, s in stats.items():
        line = (f"{m:<22} yes|COCO agrees {s['recall_coco_agrees']:.3f}  yes|COCO disagrees "
                f"{s['says_yes_coco_disagrees']:.3f}  by size " +
                " ".join(f"{r['recall']:.2f}" for r in s["recall_by_size"]))
        print(line)
        if "what_the_no_claims" in s:
            t = s["one_image_truly_missing"]
            print(f"{'':<22} wrong No {s['wrong_no_when_in_every_image']}: {s['what_the_no_claims']}, one-image "
                  f"position {s['single_image_claim_position']}, affirms all three "
                  f"{s['explanation_affirms_all_three']}, names the truly missing image "
                  f"{t['names_the_right_one']}/{t['names_one_image']}")
    print(f"\nNumbers: {method_dir / 'no_bias.json'}\nFigure: {args.figures_dir / args.method / 'no_bias.png'}")


if __name__ == "__main__":
    main()
