"""
hidden state 꺼내기와 비교: 모델 안에서 층마다 토큰별로 만들어지는 벡터
"""
import torch


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
        # out.hidden_states는 층마다 (1, 토큰 수, 벡터 크기)인 tuple이다. 입력이 하나이므로 맨 앞 차원을 없애고 쌓는다
        "hidden_states": torch.stack([layer[0] for layer in out.hidden_states]).float().cpu(),
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
