"""
pairs.json의 pair마다 변형 6개(PPP, PPN, PNN, NNN, NPP, PNP)를 펼쳐 cases.jsonl로 저장
python scripts/build_cases.py
"""
import json

from build_pairs import PAIRS_JSON
from build_pools import PROJECT_ROOT

CASES_JSONL = PROJECT_ROOT / "data" / "cases" / "cases.jsonl"
VARIANTS = {
    "PPP": [("p", 0), ("p", 1), ("p", 2)],
    "PPN": [("p", 0), ("p", 1), ("n", 0)],
    "PNN": [("p", 0), ("n", 1), ("n", 0)],
    "NNN": [("n", 2), ("n", 1), ("n", 0)],
    "NPP": [("n", 0), ("p", 1), ("p", 2)],
    "PNP": [("p", 0), ("n", 0), ("p", 2)],
}

MODES = {"PPP": ["A"], "PPN": ["A", "B"], "PNN": ["A"], "NNN": ["A"], "NPP": ["B"], "PNP": ["B"]}


def make_cases(pair: dict) -> list[dict]:
    """pair 하나를 변형 6개로 펼친다. 질문이 '3장 모두에 있나'이므로 PPP만 Yes"""
    cases = []
    for variant, slots in VARIANTS.items():
        cases.append({
            "case_id": f"{pair['group_id']}_{pair['pair_id']}_{variant}",
            "group_id": pair["group_id"],
            "pair_id": pair["pair_id"],
            "category_id": pair["category_id"],
            "object": pair["object"],
            "variant": variant,
            "modes": MODES[variant],
            "image_ids": [pair[kind][i] for kind, i in slots],
            "label": "Yes" if variant == "PPP" else "No",
        })
    return cases


def main() -> None:
    with open(PAIRS_JSON) as f:
        pairs = json.load(f)

    cases = []
    for pair in pairs:
        cases.extend(make_cases(pair))

    with open(CASES_JSONL, "w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")

    n_yes = sum(case["label"] == "Yes" for case in cases)
    n_mode_a = sum("A" in case["modes"] for case in cases)
    n_mode_b = sum("B" in case["modes"] for case in cases)
    print(f"[ok] {len(cases)} cases (Yes {n_yes} / No {len(cases) - n_yes}), mode A {n_mode_a}, mode B {n_mode_b}")
    print(f"[ok] -> {CASES_JSONL}")


if __name__ == "__main__":
    main()
