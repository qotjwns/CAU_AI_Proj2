"""
pair 기록 파일 저장과, 이어서 실행할 때의 일관성 확인

실행 폴더의 구성:
    meta.json         실행 조건
    results.jsonl     끝난 pair마다 한 줄. 파일 이름과 내용의 해시가 들어 있다
    pairs/<pair>.pt   pair마다 파일 하나 (내용은 mivlm.analysis.pair_record.build_pair_record)
pair 하나가 끝나는 순서: 임시 파일에 쓴다 → <pair>.pt로 이름을 바꾼다 → results.jsonl에 한 줄을 쓴다.
그래서 results.jsonl에 적힌 pair는 완전한 파일이 있어야 하고, 어디서 끊겨도 같은 명령으로 다시 실행하면 이어진다.
"""
import hashlib
from pathlib import Path

import torch

from mivlm.run_files import read_results

PAIRS_DIR = "pairs"


def save_pair_record(run_dir: Path, record: dict) -> dict:
    """record를 run_dir/pairs/<pair>.pt로 저장하고, results.jsonl에 적을 파일 정보를 돌려준다

    임시 파일에 다 쓴 다음 이름을 바꾼다. 이름 바꾸기는 한 번에 일어나므로, 쓰는 도중에 끊겨도 <pair>.pt는
    없거나 완전한 파일이다. 같은 이름의 파일이 이미 있으면 새 파일로 바뀐다.
    반환: {"file": "pairs/<pair>.pt", "sha256": 파일 내용의 해시}
    """
    path = run_dir / PAIRS_DIR / f"{record['pair']}.pt"
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(record, temporary)
    temporary.replace(path)
    return {"file": str(path.relative_to(run_dir)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check_saved_pairs(run_dir: Path) -> None:
    """이어서 실행하기 전에 저장된 결과가 서로 맞는지 확인한다. 맞지 않으면 RuntimeError

    results.jsonl에 적힌 pair마다: 파일이 있어야 하고, 내용의 해시가 저장할 때와 같아야 한다.
    쓰다가 끊겨서 남은 임시 파일(pairs/*.tmp)은 지운다. 그 pair는 results.jsonl에 없으므로 다시 실행된다.
    파일만 있고 results.jsonl에 없는 pair(이름을 바꾼 직후에 끊긴 경우)도 다시 실행되어 새 파일로 바뀐다.
    """
    for leftover in (run_dir / PAIRS_DIR).glob("*.tmp"):
        leftover.unlink()

    if not (run_dir / "results.jsonl").exists():
        return
    for result in read_results(run_dir):
        path = run_dir / result["file"]
        if not path.exists():
            raise RuntimeError(f"results.jsonl에는 {result['pair']}가 끝났다고 적혀 있는데 {path}가 없음. "
                               f"results.jsonl에서 그 줄을 지우면 그 pair만 다시 실행된다")
        if hashlib.sha256(path.read_bytes()).hexdigest() != result["sha256"]:
            raise RuntimeError(f"{path}의 내용이 저장할 때와 다름. "
                               f"파일과 results.jsonl의 그 줄을 지우면 그 pair만 다시 실행된다")


def load_pair_records(run_dir: Path):
    """실행 폴더의 pair 기록을 results.jsonl에 적힌 순서대로 하나씩 읽어 준다 (기록의 내용은 build_pair_record 참고)

    list로 한꺼번에 돌려주지 않고 하나씩 준다. 기록 하나가 수 MB라서 pair 400개를 모두 메모리에 올려 둘 필요가 없다.
    쓰는 쪽: for record in load_pair_records(run_dir): ...
    """
    for result in read_results(run_dir):
        yield torch.load(run_dir / result["file"], map_location="cpu", weights_only=True)
