"""
hidden state 꺼내기와 비교: 모델 안에서 층마다 토큰별로 만들어지는 벡터
position id 꺼내기: 모델이 각 토큰의 위치로 쓰는 번호
"""
import torch

# "같다"의 기준: Relative L2(차이의 크기 / 원래 크기)가 이 값보다 작으면 같은 벡터로 본다.
# 반올림 오차 수준의 차이는 넘기고, 앞의 이미지가 달라서 생기는 실제 차이는 걸러내도록 그 둘 사이의 값으로 정했다.
# 이 기준을 넘는 device와 dtype의 조합에서는 벡터 비교 결과를 그대로 쓸 수 없다 (그때는 float32로 돌린다)
SAME_WITHIN = 1e-3


def extract_hidden_states(model, inputs) -> dict:
    """입력 하나를 forward 한 번 돌려 모든 층의 hidden state와 답변 위치의 logit을 돌려준다 (답은 생성하지 않는다)

    inputs: prepare_inputs의 결과 (입력 하나). 모델의 device로 옮겨지고, 넘겨준 inputs 자체가 바뀐다
    hidden_states: shape (층 수 + 1, 토큰 수, 벡터 크기), float32, CPU
        [0]은 첫 층에 들어가기 전의 벡터다. 글자 토큰은 embedding, 이미지 토큰은 vision encoder의 출력이다.
        [k]는 k번째 층의 출력이다. 단, 마지막 항목은 마지막 층의 출력에 최종 norm을 적용한 값이다 (transformers의 동작).
        구간이 (start, end)인 이미지의 벡터들은 hidden_states[k, start:end], 답변 위치의 층별 벡터는 hidden_states[:, -1]이다.
    logits: 답변 위치(마지막 토큰)의 logit, shape (vocab 크기,), float32, CPU. infer가 Yes/No를 읽는 첫 생성 위치에 해당한다
    """
    with torch.no_grad():  # generate와 달리 직접 forward 하면 gradient 계산용 값이 쌓이므로 끈다
        out = model(**inputs.to(model.device), output_hidden_states=True)
    return {
        # out.hidden_states는 층마다 (1, 토큰 수, 벡터 크기)인 tuple이다. 입력이 하나이므로 맨 앞 차원을 없애고 쌓는다.
        # 층마다 먼저 CPU로 옮긴 다음 float32로 바꾸고 쌓는다. device에서 쌓고 바꾸면 전체 크기의 텐서가 device에
        # 두 번 더 잡힌다. bf16 → float32는 값이 그대로 옮겨지므로 어디서 바꾸든 결과는 같다
        "hidden_states": torch.stack([layer[0].cpu().float() for layer in out.hidden_states]),
        "logits": out.logits[0, -1].float().cpu(),
    }


def compare_hidden_states(a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
    """같은 이미지의 벡터를 두 조건에서 비교한다

    a, b: shape (층 수, 토큰 수, 벡터 크기), extract_hidden_states가 돌려준 float32.
        같은 이미지라서 토큰 수가 같고, t번째 토큰끼리 짝이 된다
    token_cosine: 토큰끼리의 cosine similarity, shape (층 수, 토큰 수). 토큰 하나하나가 얼마나 같은 방향인가.
        층별 평균은 .mean(dim=1)
    cosine_of_means: 토큰을 먼저 평균 낸 벡터끼리의 cosine similarity, shape (층 수,). 이미지 전체의 대표 벡터가
        얼마나 같은가. 토큰별 차이가 평균에서 서로 상쇄되면 token_cosine의 평균보다 높게 나온다
    relative_l2: 차이의 크기 / a의 크기, shape (층 수,). cosine이 못 보는 벡터 크기의 차이까지 잡는다
    """
    return {
        "token_cosine": torch.nn.functional.cosine_similarity(a, b, dim=-1),
        "cosine_of_means": torch.nn.functional.cosine_similarity(a.mean(dim=1), b.mean(dim=1), dim=-1),
        "relative_l2": (a - b).flatten(1).norm(dim=1) / a.flatten(1).norm(dim=1),
    }


def position_ids(model, inputs) -> torch.Tensor:
    """모델이 이 입력의 각 토큰에 쓰는 position id를 돌려준다: shape (3, 토큰 수), int64, CPU

    3은 (시간, 세로, 가로) 축이다. 글자 토큰은 세 축의 값이 같다.
    이미지 토큰은 격자 안의 위치를 따른다: 시간 축은 이미지 전체가 같은 값이고, 세로 축은 행마다, 가로 축은 열마다 1씩 커진다.
    이미지 뒤의 토큰은 그 이미지의 토큰 수가 아니라 max(행 수, 열 수)만큼 건너뛴 번호를 받는다.
    그래서 position id는 토큰 인덱스와 다르고, 토큰 인덱스가 같아도 position id가 같다고 볼 수 없다.

    Qwen2.5-VL 전용이다. forward가 position id를 만들 때 부르는 함수를 같은 값으로 부른다.
    inputs: prepare_inputs의 결과 (입력 하나)
    """
    ids = model.model.compute_3d_position_ids(
        input_ids=inputs["input_ids"],
        image_grid_thw=inputs["image_grid_thw"],
        video_grid_thw=None,
        inputs_embeds=None,
        attention_mask=inputs["attention_mask"],
        past_key_values=None,
        mm_token_type_ids=inputs["mm_token_type_ids"],
    )  # (3, 1, 토큰 수)
    return ids[:, 0].cpu()
