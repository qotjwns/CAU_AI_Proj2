"""
분해 실험 B: 각각 이미지 추론 후 and (클래스 20개 x 16장 = 320번) 
결과는 outputs/predictions/02_decomposition/b_single_image/<모델>[_float32][_pilot]/에 저장된다
python experiments/02_decomposition/run_b_single_image.py --pilot     # 클래스마다 P 1장, N 1장, 40개 (Mac)
python experiments/02_decomposition/run_b_single_image.py             # 전체 320개 (GPU 서버)
"""
from mivlm.data.cases import items_for_cases, load_cases, load_pools, pilot_cases, single_image_items
from mivlm.models.inference import SINGLE_IMAGE_QUESTION, single_image_messages
from mivlm.runner import parse_run_args, run_inference


def main() -> None:
    args = parse_run_args("분해 실험 B: 이미지 1장씩 추론")
    items = single_image_items(load_pools(args.pool))
    if args.pilot and args.pool is None:
        # 항목이 클래스마다 P 8장, N 8장 순서라서, 8개마다 하나씩 고르면 클래스별 첫 P와 첫 N이 된다
        items = items[::8]
    elif args.pilot:
        # 새 풀: A 파일럿 케이스(pilot_cases)에 쓰인 이미지들. A와 B를 같은 케이스로 비교할 수 있다
        items = items_for_cases(items, pilot_cases(load_cases(args.pool), args.pool))
    run_inference("02_decomposition/b_single_image", args, items, "item_id", single_image_messages, SINGLE_IMAGE_QUESTION)


if __name__ == "__main__":
    main()
