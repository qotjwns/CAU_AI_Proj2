"""
STEP 1 baseline: 케이스를 추론해 outputs/predictions/01_baseline/<모델 이름>/에 결과를 저장
python experiments/01_baseline/run.py --pilot     # 클래스별 1개, 20케이스 (Mac)
python experiments/01_baseline/run.py             # 전체 2,400케이스 (GPU 서버)
python experiments/01_baseline/run.py --dtype float32     # float32로 실행, bf16 결과와 비교용 (GPU 서버)
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

from mivlm.config import CASES_JSONL, PROJECT_ROOT, load_models_config
from mivlm.data.cases import load_cases, pilot_cases
from mivlm.models.inference import GENERATION_KWARGS, QUESTION, case_messages, infer
from mivlm.models.loading import load_model

EXPERIMENT = "01_baseline"


def package_version(name: str) -> str | None:
    """설치된 패키지 버전. 설치돼 있지 않으면 None"""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    models = load_models_config()
    parser = argparse.ArgumentParser(description="STEP 1 baseline 추론")
    parser.add_argument("--model", choices=list(models), default="qwen2.5-vl-3b")
    parser.add_argument("--pilot", action="store_true", help="클래스별 1개씩 20케이스만 실행")
    parser.add_argument("--dtype", choices=["float32"], help="지정하지 않으면 device 기본값 (CUDA, MPS는 bfloat16)")
    args = parser.parse_args()

    cases = load_cases()
    if args.pilot:
        cases = pilot_cases(cases)
    # 결과 폴더 이름: <모델>[_float32][_pilot]
    run_name = args.model + (f"_{args.dtype}" if args.dtype else "") + ("_pilot" if args.pilot else "")
    run_dir = PROJECT_ROOT / "outputs" / "predictions" / EXPERIMENT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    repo_id = models[args.model]
    model, processor = load_model(args.model, getattr(torch, args.dtype) if args.dtype else None)
    meta = {
        "experiment": EXPERIMENT,
        "pilot": args.pilot,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "n_cases": len(cases),
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
            "torchvision": package_version("torchvision"),
            # None이면 이미지를 torchvision으로 디코딩한 것이다 (0.16 이상이 있으면 torchcodec을 쓴다)
            "torchcodec": package_version("torchcodec"),
        },
        # repetition_penalty는 따로 넘기지 않아서 모델 기본값이 적용된다
        "generation": {**GENERATION_KWARGS, "repetition_penalty": model.generation_config.repetition_penalty},
        "question": QUESTION,
        "cases_sha256": hashlib.sha256(CASES_JSONL.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "platform": platform.platform(),
        },
        "git": {"commit": git("rev-parse", "HEAD"), "dirty": git("status", "--porcelain") != ""},
    }

    # 이어서 돌릴 때: 기존 실행과 조건이 같아야 한다 (시작 시각, 명령 문자열, git은 비교하지 않는다)
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            old_meta = json.load(f)
        not_compared = ("started_at", "command", "git")
        changed = [key for key in meta if key not in not_compared and meta[key] != old_meta.get(key)]
        if changed:
            raise RuntimeError(f"{run_dir}의 기존 실행과 조건이 다름: {changed}. 폴더를 지우고 다시 실행")
    else:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    results_path = run_dir / "results.jsonl"
    done = set()
    if results_path.exists():
        with open(results_path) as f:
            done = {json.loads(line)["case_id"] for line in f}
    todo = [case for case in cases if case["case_id"] not in done]
    print(f"{len(cases)} cases, {len(done)} done, {len(todo)} to run -> {run_dir}")

    start = time.time()
    with open(results_path, "a") as f:
        for case in tqdm(todo, desc=args.model):
            result = infer(model, processor, case_messages(case))
            # 케이스마다 한 줄 쓰고 바로 디스크에 내보낸다. 중간에 끊겨도 여기까지의 결과는 남는다
            f.write(json.dumps({"case_id": case["case_id"], **result}) + "\n")
            f.flush()

    if todo:
        elapsed = time.time() - start
        print(f"[ok] {len(todo)} cases in {elapsed:.0f}s ({elapsed / len(todo):.2f}s per case)")


if __name__ == "__main__":
    main()
