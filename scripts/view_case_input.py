"""
케이스 하나를 실제로 추론하고, 모델에 들어간 입력(이미지 3장, 토큰 위치)과 답·logit을 그림 한 장으로 저장
python scripts/view_case_input.py group_01_person_1_PPN
"""
import argparse
import json

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from mivlm.config import INSTANCES_JSON, PROJECT_ROOT
from mivlm.data.cases import load_cases
from mivlm.data.image import image_path
from mivlm.models.inference import ALL_IMAGES_QUESTION, case_messages, infer
from mivlm.models.loading import load_model

MODEL_NAME = "qwen2.5-vl-3b"
OUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "case_inputs"


def restore_image(patches: torch.Tensor, grid_h: int, grid_w: int, image_processor) -> torch.Tensor:
    """pixel_values의 이미지 한 장 분량 (grid_h*grid_w, 1176)을 그림 (높이, 너비, 3), 값 0~1로 되돌린다

    processor가 한 일(정규화 → 14px 패치로 자르기 → 2x2씩 묶어 한 줄로 펴기)을 거꾸로 한다.
    """
    patch, merge, temporal = image_processor.patch_size, image_processor.merge_size, image_processor.temporal_patch_size
    x = patches.reshape(grid_h // merge, grid_w // merge, merge, merge, 3, temporal, patch, patch)
    x = x[:, :, :, :, :, 0]
    x = x.permute(4, 0, 2, 5, 1, 3, 6).reshape(3, grid_h * patch, grid_w * patch)
    mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
    std = torch.tensor(image_processor.image_std).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1).permute(1, 2, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="케이스 하나의 입력과 모델 답을 그림으로 저장")
    parser.add_argument("case_id")
    args = parser.parse_args()

    cases_by_id = {case["case_id"]: case for case in load_cases()}
    case = cases_by_id[args.case_id]
    with open(INSTANCES_JSON) as f:
        annotations = json.load(f)["annotations"]

    model, processor = load_model(MODEL_NAME)
    messages = case_messages(case)
    result = infer(model, processor, messages)

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    grid = inputs["image_grid_thw"].tolist() 
    patches_per_image = inputs["pixel_values"].split([h * w for _, h, w in grid])
    image_token_positions = (inputs["input_ids"][0] == processor.image_token_id).nonzero().flatten().tolist()

    print(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    print(f"case: {case}")
    print(f"result: {result}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    first = 0 
    for slot, image_id in enumerate(case["image_ids"]):
        letter = case["variant"][slot]
        _, grid_h, grid_w = grid[slot]
        n_tokens = grid_h * grid_w // processor.image_processor.merge_size**2
        start, end = image_token_positions[first], image_token_positions[first + n_tokens - 1]
        first += n_tokens
        print(f"slot {slot + 1}: {letter} image {image_id}, grid {grid_h}x{grid_w}, tokens {start}-{end} ({n_tokens})")

        top = axes[0, slot]
        top.imshow(plt.imread(image_path(image_id)), cmap="gray")
        for ann in annotations:
            if ann["image_id"] == image_id and ann["category_id"] == case["category_id"]:
                x, y, w, h = ann["bbox"]
                top.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2))
        top.set_title(f"slot {slot + 1}: {letter}  image {image_id}  (original file)")
        top.axis("off")

        bottom = axes[1, slot]
        bottom.imshow(restore_image(patches_per_image[slot], grid_h, grid_w, processor.image_processor).numpy())
        bottom.set_title(f"model input: tokens {start}-{end} ({n_tokens})")
        bottom.axis("off")

    fig.suptitle(
        f"{case['case_id']}   object: {case['object']}   label: {case['label']}\n"
        f"answer: {result['raw_response']!r}   yes {result['yes_logit']:.3f} / no {result['no_logit']:.3f}   "
        f"margin {result['logit_margin']:+.3f}   input tokens {result['input_token_count']}"
    )
    fig.text(0.5, 0.02, ALL_IMAGES_QUESTION.format(object=case["object"]), ha="center")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{case['case_id']}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[ok] {out_path}")


if __name__ == "__main__":
    main()
