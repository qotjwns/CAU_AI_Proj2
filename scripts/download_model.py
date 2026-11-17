"""
configs/models.yaml에 등록된 모델을 Hugging Face 캐시로 다운로드
python scripts/download_model.py --help            # 등록된 모델 이름 확인
python scripts/download_model.py qwen2.5-vl-3b
"""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

from mivlm.config import load_models_config


def main() -> None:
    models = load_models_config()

    parser = argparse.ArgumentParser(description="configs/models.yaml안에 있는 모델을 다운로드")
    parser.add_argument("name", choices=list(models))
    args = parser.parse_args()

    repo_id = models[args.name]
    snapshot_dir = Path(snapshot_download(repo_id))

    print(f"[ok] {args.name}: {repo_id}")
    print(f"[ok] commit: {snapshot_dir.name}")
    print(f"[ok] path: {snapshot_dir}")


if __name__ == "__main__":
    main()
