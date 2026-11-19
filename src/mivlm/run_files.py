"""
실행 폴더 읽기: outputs/predictions/<실험>/<실행 이름>/의 meta.json과 results.jsonl

두 파일을 쓰는 쪽은 mivlm.runner다. 여기는 읽기만 하고 모델 관련 패키지를 import하지 않는다.
그래서 저장된 결과만 읽는 코드(지표, 요약, 그림)가 모델 없이 가볍게 쓸 수 있다.
"""
import json
from pathlib import Path


def read_meta(run_dir: Path) -> dict:
    """실행 폴더의 meta.json을 읽는다 (실행 조건. 내용은 mivlm.runner.build_meta 참고)"""
    with open(run_dir / "meta.json") as f:
        return json.load(f)


def read_results(run_dir: Path) -> list[dict]:
    """실행 폴더의 results.jsonl을 읽는다 (한 줄에 결과 하나)"""
    with open(run_dir / "results.jsonl") as f:
        return [json.loads(line) for line in f]
