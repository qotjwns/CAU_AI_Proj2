"""
hidden state 꺼내기: 모델 안에서 층마다 토큰별로 만들어지는 벡터
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
