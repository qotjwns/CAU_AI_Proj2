"""configs/models.yaml에 등록된 모델을 Hugging Face 캐시로 다운로드"""
import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

MODELS_YAML = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"


def main() -> None:
    with open(MODELS_YAML) as f:
        models = yaml.safe_load(f)["models"]

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
