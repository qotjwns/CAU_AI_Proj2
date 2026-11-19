"""
pair 하나의 저장용 기록 만들기: 변형 6개를 forward 해서 모은 값에서 케이스별 값과 12개 비교를 뽑는다

흐름: observe_case(케이스마다 forward) → build_pair_record(케이스 6개의 값 + 비교 12개) → pair_summary(요약)

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

from mivlm.analysis.hidden_states import compare_hidden_states, extract_hidden_states, position_ids
from mivlm.models.inference import case_messages, image_token_spans, prepare_inputs, yes_no_token_ids

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


def check_image_layout(case_id: str, token_spans: list[tuple[int, int]], image_grid_thw: list[list[int]],
                       ids: torch.Tensor, merge_size: int) -> None:
    """이미지 토큰 구간, 격자, position id가 서로 맞는지 확인한다. 맞지 않으면 ValueError

    ids: shape (3, 토큰 수). position_ids()의 결과
    merge_size: 패치를 가로세로 몇 개씩 합쳐 토큰 하나로 만드는가 (Qwen2.5-VL은 2)
    이미지마다 확인하는 것:
        토큰 수 = (세로 패치 수 / merge_size) x (가로 패치 수 / merge_size)
        position id: 시간 축은 전부 첫 토큰의 값, 세로 축은 행마다, 가로 축은 열마다 1씩 커진다 (행 우선 순서)
    이것이 성립하면 이미지 토큰의 position id는 첫 토큰의 값(position_start)과 격자만으로 다시 만들 수 있다.
    그래서 position id 전체는 저장하지 않고 position_start만 저장한다.
    """
    if len(token_spans) != len(image_grid_thw):
        raise ValueError(f"{case_id}: 토큰 구간은 {len(token_spans)}개인데 격자는 {len(image_grid_thw)}개다")
    for slot, ((start, end), (_, grid_h, grid_w)) in enumerate(zip(token_spans, image_grid_thw), start=1):
        rows, columns = grid_h // merge_size, grid_w // merge_size
        if end - start != rows * columns:
            raise ValueError(f"{case_id} 자리 {slot}: 토큰 수 {end - start}가 격자 {rows} x {columns}와 다르다")
        first = ids[0, start]
        expected = torch.stack([
            first.expand(rows * columns),
            first + torch.arange(rows).repeat_interleave(columns),
            first + torch.arange(columns).repeat(rows),
        ])
        if not torch.equal(ids[:, start:end], expected):
            raise ValueError(f"{case_id} 자리 {slot}: position id가 격자 {rows} x {columns}의 행 우선 순서와 다르다")


def observe_case(model, processor, case: dict) -> dict:
    """케이스 하나를 forward 해서 observation을 만든다 (observation의 내용은 case_record 참고)

    baseline(infer)과 같은 메시지와 같은 입력을 쓴다. 답은 생성하지 않는다.
    forward 전에 이미지 수, 토큰 구간, 격자, position id가 서로 맞는지 확인한다.
    hidden_states가 케이스 하나에 수백 MB이므로, 돌려받은 observation은 비교가 끝나면 버린다.
    """
    inputs = prepare_inputs(processor, case_messages(case))
    token_spans = image_token_spans(inputs["input_ids"][0], processor.image_token_id)
    image_grid_thw = inputs["image_grid_thw"].tolist()
    ids = position_ids(model, inputs)
    if len(token_spans) != len(case["image_ids"]):
        raise ValueError(f"{case['case_id']}: 이미지는 {len(case['image_ids'])}장인데 토큰 구간은 {len(token_spans)}개다")
    check_image_layout(case["case_id"], token_spans, image_grid_thw, ids, processor.image_processor.merge_size)

    extracted = extract_hidden_states(model, inputs)
    yes_id, no_id = yes_no_token_ids(processor.tokenizer)
    return {
        "case_id": case["case_id"],
        "image_ids": case["image_ids"],
        "token_spans": token_spans,
        "image_grid_thw": image_grid_thw,
        "position_ids": ids,
        "hidden_states": extracted["hidden_states"],
        "yes_logit": extracted["logits"][yes_id].item(),
        "no_logit": extracted["logits"][no_id].item(),
    }


def case_record(observation: dict) -> dict:
    """케이스 하나의 저장용 기록

    observation: 케이스 하나를 forward 해서 모은 값 (observe_case의 결과)
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


def pair_summary(record: dict) -> dict:
    """results.jsonl에 적을 pair 요약. JSON으로 쓸 수 있는 값만 담아서, .pt 파일을 열지 않고도 확인할 수 있게 한다

    cases: 케이스별 logit과 토큰 수. baseline 결과와 logit margin을 비교하는 데 쓴다
    same_token_index, same_position_ids: 비교 12개의 값 (COMPARISONS 순서). 서로 다른 조건이라 따로 적는다
    same_prefix_max_relative_l2: same_prefix 기준선(앞부분이 완전히 같은 두 입력)의 Relative L2 중 가장 큰 층의 값
    same_prefix_identical: 그 값이 0인가, 곧 기준선의 벡터가 모든 층에서 완전히 같은가.
        같아야 한다는 것은 예측이고, 실제로 같은지는 기기와 dtype에 따라 다를 수 있어서 측정값과 함께 적는다
    """
    (same_prefix,) = [comparison for comparison in record["comparisons"] if comparison["family"] == "same_prefix"]
    largest = same_prefix["relative_l2"].max().item()
    case_keys = ["case_id", "yes_logit", "no_logit", "logit_margin", "input_token_count"]
    return {
        "cases": {variant: {key: case[key] for key in case_keys} for variant, case in record["cases"].items()},
        "same_token_index": [comparison["same_token_index"] for comparison in record["comparisons"]],
        "same_position_ids": [comparison["same_position_ids"] for comparison in record["comparisons"]],
        "same_prefix_max_relative_l2": largest,
        "same_prefix_identical": largest == 0,
    }
