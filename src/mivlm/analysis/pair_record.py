"""
pair 하나의 저장용 기록 만들기: 변형 6개를 forward 해서 모은 값에서 케이스별 값과 12개 비교를 뽑는다

pair의 구성 (cases.jsonl의 모든 pair가 같다):
    PPP = [P1, P2, P3]   PPN = [P1, P2, N1]   PNP = [P1, N1, P3]
    NPP = [N1, P2, P3]   PNN = [P1, N2, N1]   NNN = [N3, N2, N1]

위치를 가리키는 값이 두 가지이고 서로 다르다. 그래서 이름을 나눠 쓴다.
    토큰 인덱스 (token_span, same_token_index): 모델 입력에서 몇 번째 토큰인가. 앞에 놓인 토큰의 개수와 같다
    position id (position_start, same_position_ids): 모델이 위치로 쓰는 번호. 이미지 하나를 지나면 토큰 수가 아니라
        max(행 수, 열 수)만큼만 커진다. 그래서 토큰 인덱스가 달라도 position id는 같을 수 있고, 그 반대도 있다
position id가 같다는 것은 모델이 보는 위치가 같다는 뜻일 뿐이다. 그것만으로 남은 차이가 앞 이미지의 내용 때문이라고
단정할 수 없다 (앞에 놓인 토큰의 개수, 앞 이미지 토큰들의 position id 배치가 다를 수 있다).
"""
import torch

from mivlm.analysis.hidden_states import compare_hidden_states

# 비교 12개: (family, 이미지 이름, (변형, 자리), (변형, 자리)). 자리는 1부터 센다.
# 순서와 family는 저장된 결과를 읽는 쪽이 기대는 값이므로 바꾸지 않는다
COMPARISONS = [
    # slot_change: 같은 N1이 자리 1, 2, 3으로 옮겨 간다
    ("slot_change", "N1", ("NPP", 1), ("PNP", 2)),
    ("slot_change", "N1", ("NPP", 1), ("PPN", 3)),
    ("slot_change", "N1", ("PNP", 2), ("PPN", 3)),
    # n1_at_slot_3: N1은 자리 3에 그대로 있고, 앞의 두 자리가 P인지 N인지만 달라진다
    ("n1_at_slot_3", "N1", ("PPN", 3), ("PNN", 3)),  # 자리 2: P2 → N2
    ("n1_at_slot_3", "N1", ("PNN", 3), ("NNN", 3)),  # 자리 1: P1 → N3
    ("n1_at_slot_3", "N1", ("PPN", 3), ("NNN", 3)),  # 자리 1, 2 모두
    # same_slot: 이미지와 자리는 그대로이고 앞의 이미지만 달라진다
    ("same_slot", "P2", ("PPP", 2), ("NPP", 2)),  # 자리 1: P1 → N1
    ("same_slot", "N2", ("PNN", 2), ("NNN", 2)),  # 자리 1: P1 → N3
    ("same_slot", "P3", ("PPP", 3), ("PNP", 3)),  # 자리 2: P2 → N1
    ("same_slot", "P3", ("PPP", 3), ("NPP", 3)),  # 자리 1: P1 → N1
    ("same_slot", "P3", ("PNP", 3), ("NPP", 3)),  # 자리 1, 2 모두
    # same_prefix: 앞부분이 완전히 같다. 벡터도 같아야 하는 기준선
    ("same_prefix", "P2", ("PPP", 2), ("PPN", 2)),
]


def case_record(observation: dict) -> dict:
    """케이스 하나의 저장용 기록

    observation: 케이스 하나를 forward 해서 모은 값
        case_id: str
        image_ids: 이미지 3장의 id
        token_spans: 이미지 3장의 토큰 인덱스 구간 [(start, end), ...]. end는 포함하지 않는다
        image_grid_thw: 이미지마다 [시간, 세로 패치 수, 가로 패치 수]
        position_ids: shape (3, 토큰 수). position_ids()의 결과
        hidden_states: shape (층 수 + 1, 토큰 수, 벡터 크기), float32. extract_hidden_states()의 결과
        yes_logit, no_logit: 답변 위치의 logit에서 "Yes", "No" 토큰의 값

    돌려주는 기록에서:
        position_starts: 이미지마다 첫 토큰의 position id (첫 토큰은 세 축의 값이 같다)
        pre_answer_hidden: 입력의 마지막 토큰 위치의 층별 벡터, shape (층 수 + 1, 벡터 크기).
            생성된 Yes/No 토큰의 벡터가 아니라, 첫 답변 토큰을 예측하는 위치의 벡터다
    """
    hidden_states = observation["hidden_states"]
    return {
        "case_id": observation["case_id"],
        "image_ids": observation["image_ids"],
        "token_spans": [[start, end] for start, end in observation["token_spans"]],
        "position_starts": [observation["position_ids"][0, start].item() for start, _ in observation["token_spans"]],
        "image_grid_thw": observation["image_grid_thw"],
        "input_token_count": hidden_states.shape[1],
        "yes_logit": observation["yes_logit"],
        "no_logit": observation["no_logit"],
        "logit_margin": observation["yes_logit"] - observation["no_logit"],
        # clone을 하지 않으면 전체 hidden state의 일부를 가리키는 텐서가 되어, 저장할 때 전체가 함께 저장된다
        "pre_answer_hidden": hidden_states[:, -1].clone(),
    }


def comparison_record(family: str, image: str, a_ref: tuple[str, int], b_ref: tuple[str, int],
                      observations: dict[str, dict]) -> dict:
    """비교 하나의 저장용 기록: 같은 이미지가 두 입력에 놓였을 때 그 이미지 토큰의 벡터를 비교한다

    a_ref, b_ref: (변형, 자리). 자리는 1부터 센다
    observations: {변형: observation}. observation의 내용은 case_record 참고

    돌려주는 기록에서:
        a, b: 각 입력에서 그 이미지의 자리, 토큰 인덱스 구간(token_span), 첫 토큰의 position id(position_start)
        same_token_index: 두 입력에서 토큰 인덱스 구간이 같은가
        same_position_ids: 두 입력에서 이 이미지 토큰들의 position id가 세 축 모두 같은가
        token_cosine: shape (층 수 + 1, 이 이미지의 토큰 수)
        cosine_of_means, relative_l2: shape (층 수 + 1,)
    토큰을 평균 낸 벡터 자체는 담지 않는다.
    """
    (a_variant, a_slot), (b_variant, b_slot) = a_ref, b_ref
    a, b = observations[a_variant], observations[b_variant]
    a_image_id, b_image_id = a["image_ids"][a_slot - 1], b["image_ids"][b_slot - 1]
    (a_start, a_end), (b_start, b_end) = a["token_spans"][a_slot - 1], b["token_spans"][b_slot - 1]

    where = f"{image}: {a_variant} 자리 {a_slot} vs {b_variant} 자리 {b_slot}"
    if a_image_id != b_image_id:
        raise ValueError(f"{where}: 같은 이미지가 아니다 ({a_image_id}, {b_image_id})")
    if a_end - a_start != b_end - b_start:
        raise ValueError(f"{where}: 토큰 수가 다르다 ({a_end - a_start}, {b_end - b_start})")

    a_positions = a["position_ids"][:, a_start:a_end]  # (3, 이 이미지의 토큰 수)
    b_positions = b["position_ids"][:, b_start:b_end]
    return {
        "family": family,
        "image": image,
        "image_id": a_image_id,
        "a": {"variant": a_variant, "slot": a_slot, "token_span": [a_start, a_end],
              "position_start": a_positions[0, 0].item()},
        "b": {"variant": b_variant, "slot": b_slot, "token_span": [b_start, b_end],
              "position_start": b_positions[0, 0].item()},
        "same_token_index": a_start == b_start,
        "same_position_ids": torch.equal(a_positions, b_positions),
        **compare_hidden_states(a["hidden_states"][:, a_start:a_end], b["hidden_states"][:, b_start:b_end]),
    }


def build_pair_record(pair: str, observations: dict[str, dict]) -> dict:
    """pair 하나의 저장용 기록: 케이스 6개의 값과 비교 12개

    pair: pair 이름. 예: "group_05_cat_4"
    observations: {변형: observation}. 변형 6개(PPP, PPN, PNP, NPP, PNN, NNN)가 모두 있어야 한다
    """
    return {
        "pair": pair,
        "cases": {variant: case_record(observation) for variant, observation in observations.items()},
        "comparisons": [comparison_record(family, image, a_ref, b_ref, observations)
                        for family, image, a_ref, b_ref in COMPARISONS],
    }
