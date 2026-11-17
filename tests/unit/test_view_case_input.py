"""
scripts/view_case_input.py의 restore_image 단위 테스트 (모델은 로딩하지 않는다)
pytest tests/unit/test_view_case_input.py -v
"""
import torch
from transformers import Qwen2VLImageProcessor

from view_case_input import restore_image


def test_restore_image_undoes_processor_patchify():
    # 가짜 그림(값 0~1) → 정규화 → transformers의 patchify → restore_image → 처음 그림과 같아야 한다
    image_processor = Qwen2VLImageProcessor()
    image = torch.rand(3, 56, 84)  # 채널, 높이, 너비 (둘 다 28의 배수)
    mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
    std = torch.tensor(image_processor.image_std).view(3, 1, 1)

    patches, grid_h, grid_w = image_processor.patchify(
        ((image - mean) / std)[None], patch_size=14, merge_size=2, temporal_patch_size=2
    )
    restored = restore_image(patches[0], grid_h, grid_w, image_processor)

    assert (grid_h, grid_w) == (4, 6)
    assert restored.shape == (56, 84, 3)
    assert torch.allclose(restored, image.permute(1, 2, 0), atol=1e-5)
