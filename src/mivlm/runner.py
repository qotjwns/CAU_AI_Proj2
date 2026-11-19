"""
실험 실행의 공통 절차: 옵션 읽기 → 모델 로딩 → meta.json → 남은 항목 추론 → results.jsonl

실험 스크립트는 '무엇을 추론할지'(항목 목록, 항목을 메시지로 바꾸는 법, 질문 문구)만 정해서 run_inference를 부른다.
답을 생성하지 않는 실행(hidden state 저장 등)은 start_run과 run_items를 직접 부른다.
결과는 outputs/predictions/<실험>/<모델>[_float32][_pilot]/에 저장된다.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import torch
from huggingface_hub import try_to_load_from_cache
from tqdm import tqdm

from mivlm.config import CASES_JSONL, POOLS_JSON, PREDICTIONS_DIR, PROJECT_ROOT, load_models_config
from mivlm.models.inference import GENERATION_KWARGS, infer
from mivlm.models.loading import load_model

# 이어서 돌릴 때 기존 실행과 비교하지 않는 meta 항목
NOT_COMPARED = ("started_at", "command", "git")


def parse_run_args(description: str) -> argparse.Namespace:
    """모든 실험 스크립트가 같이 쓰는 옵션: --model, --pilot, --dtype"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", choices=list(load_models_config()), default="qwen2.5-vl-3b")
    parser.add_argument("--pilot", action="store_true", help="적은 수의 항목만 실행 (Mac 확인용)")
    parser.add_argument("--dtype", choices=["float32"], help="지정하지 않으면 device 기본값 (CUDA, MPS는 bfloat16)")
    return parser.parse_args()


def _package_version(name: str) -> str | None:
    """설치된 패키지 버전. 설치돼 있지 않으면 None"""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def build_meta(experiment: str, args: argparse.Namespace, n_items: int, model, processor, question: str) -> dict:
    """실행 조건 기록. 결과를 좌우하는 조건을 모두 담는다"""
    repo_id = load_models_config()[args.model]
    return {
        "experiment": experiment,
        "pilot": args.pilot,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "n_items": n_items,
        "model": {
            "name": args.model,
            "repo_id": repo_id,
            # 캐시 폴더 이름이 받은 버전의 commit hash다 (.../snapshots/<commit hash>/config.json)
            "commit": Path(try_to_load_from_cache(repo_id, "config.json")).parent.name,
            "device": str(model.device),
            "dtype": str(model.dtype),
            # 둘 다 False여야 float32 연산이 정확한 float32로 계산된다 (load_model이 끈다)
            "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32},
            "attn_implementation": model.config._attn_implementation,
        },
        "preprocessing": {
            "image_processor": type(processor.image_processor).__name__,
            "torchvision": _package_version("torchvision"),
            # None이면 이미지를 torchvision으로 디코딩한 것이다 (0.16 이상이 있으면 torchcodec을 쓴다)
            "torchcodec": _package_version("torchcodec"),
        },
        # repetition_penalty는 따로 넘기지 않아서 모델 기본값이 적용된다
        "generation": {**GENERATION_KWARGS, "repetition_penalty": model.generation_config.repetition_penalty},
        "question": question,
        "cases_sha256": hashlib.sha256(CASES_JSONL.read_bytes()).hexdigest(),
        "pools_sha256": hashlib.sha256(POOLS_JSON.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "platform": platform.platform(),
        },
        "git": {"commit": _git("rev-parse", "HEAD"), "dirty": _git("status", "--porcelain") != ""},
    }


def check_or_save_meta(run_dir: Path, meta: dict) -> None:
    """run_dir에 meta.json이 없으면 저장한다. 있으면 조건이 같은지 확인하고, 다르면 에러를 낸다

    조건이 다른 실행의 결과가 한 results.jsonl에 섞이는 것을 막는다. 기존 meta.json은 덮어쓰지 않는다.
    """
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return
    with open(meta_path) as f:
        old_meta = json.load(f)
    changed = [key for key in meta if key not in NOT_COMPARED and meta[key] != old_meta.get(key)]
    if changed:
        raise RuntimeError(f"{run_dir}의 기존 실행과 조건이 다름: {changed}. 폴더를 지우고 다시 실행")


def run_items(run_dir: Path, items: list[dict], id_key: str, infer_item) -> int:
    """items를 차례로 추론해 run_dir/results.jsonl에 한 줄씩 덧붙인다. 실행한 개수를 돌려준다

    id_key: 항목을 구분하는 필드 이름 (예: "case_id"). 결과 파일에 이미 있는 id는 건너뛴다.
    infer_item: 항목 하나를 받아 결과 dict를 돌려주는 함수
    """
    results_path = run_dir / "results.jsonl"
    done = set()
    if results_path.exists():
        with open(results_path) as f:
            done = {json.loads(line)[id_key] for line in f}
    todo = [item for item in items if item[id_key] not in done]
    print(f"{len(items)} items, {len(done)} done, {len(todo)} to run -> {run_dir}")

    start = time.time()
    with open(results_path, "a") as f:
        for item in tqdm(todo):
            # 항목마다 한 줄 쓰고 바로 디스크에 내보낸다. 중간에 끊겨도 여기까지의 결과는 남는다
            f.write(json.dumps({id_key: item[id_key], **infer_item(item)}) + "\n")
            f.flush()
    if todo:
        elapsed = time.time() - start
        print(f"[ok] {len(todo)} items in {elapsed:.0f}s ({elapsed / len(todo):.2f}s per item)")
    return len(todo)


def start_run(experiment: str, args: argparse.Namespace, n_items: int, question: str):
    """실행 폴더를 만들고, 모델을 로딩하고, meta.json을 저장하거나 기존 것과 조건이 같은지 확인한다

    experiment: outputs/predictions/ 아래의 폴더 경로 (예: "01_baseline")
    question: meta.json에 기록할 질문 문구
    반환: (run_dir, model, processor). run_dir은 outputs/predictions/<실험>/<모델>[_float32][_pilot]/
    """
    run_name = args.model + (f"_{args.dtype}" if args.dtype else "") + ("_pilot" if args.pilot else "")
    run_dir = PREDICTIONS_DIR / experiment / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args.model, getattr(torch, args.dtype) if args.dtype else None)
    check_or_save_meta(run_dir, build_meta(experiment, args, n_items, model, processor, question))
    return run_dir, model, processor


def run_inference(experiment: str, args: argparse.Namespace, items: list[dict], id_key: str, make_messages, question: str) -> None:
    """모델을 로딩하고 items를 추론해 저장한다

    make_messages: 항목 하나를 모델에 넣을 메시지로 바꾸는 함수
    나머지 인자는 start_run 참고
    """
    run_dir, model, processor = start_run(experiment, args, len(items), question)
    run_items(run_dir, items, id_key, lambda item: infer(model, processor, make_messages(item)))
