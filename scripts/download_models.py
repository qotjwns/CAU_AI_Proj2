#!/usr/bin/env python3
"""
    Download VLM checkpoints to models/
      Qwen2.5-VL-3B-Instruct/           qwen2.5-vl-3b   (from Qwen/Qwen2.5-VL-3B-Instruct, 7.5 GB)
      Qwen2.5-VL-7B-Instruct/           qwen2.5-vl-7b   (from Qwen/Qwen2.5-VL-7B-Instruct, 16.6 GB)
      InternVL2_5-4B/                   internvl2.5-4b  (from OpenGVLab/InternVL2_5-4B, 7.4 GB)
      llava-onevision-qwen2-7b-ov-hf/   llava-ov-7b     (from llava-hf/llava-onevision-qwen2-7b-ov-hf, 16.1 GB)
    python scripts/download_models.py --model qwen2.5-vl-3b llava-ov-7b  # one or more names above
    python scripts/download_models.py --model all                        # all four, ~48 GB
"""
import argparse
import fnmatch
import shutil
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError:
    sys.exit("huggingface_hub is not installed: pip install huggingface_hub")

ROOT = Path(__file__).resolve().parents[1] / "models"
MODELS = {
    "qwen2.5-vl-3b": ("Qwen/Qwen2.5-VL-3B-Instruct", "66285546d2b821cf421d4f5eb2576359d3770cd3"),
    "qwen2.5-vl-7b": ("Qwen/Qwen2.5-VL-7B-Instruct", "cc594898137f460bfe9f0759e9844b3ce807cfb5"),
    "internvl2.5-4b": ("OpenGVLab/InternVL2_5-4B", "2cf4a8158bbc40d35015e7c63b527890de4d27b3"),
    "llava-ov-7b": ("llava-hf/llava-onevision-qwen2-7b-ov-hf", "0d50680527681998e456c7b78950205bedd8a068"),
}
IGNORE = ["examples/*", "runs/*", "*.png", "*.jpg", "*.mp4"]
GB = 1e9


def expected_files(api, repo, rev):
    info = api.model_info(repo, revision=rev, files_metadata=True)
    return {s.rfilename: s.size for s in info.siblings
            if not any(fnmatch.fnmatch(s.rfilename, p) for p in IGNORE)}


def missing_bytes(dest, files):
    """Bytes still to fetch: files that are absent or have the wrong size."""
    return sum(size for name, size in files.items()
               if not (dest / name).is_file() or (dest / name).stat().st_size != size)


def download(api, name):
    repo, rev = MODELS[name]
    dest = ROOT / repo.split("/")[1]
    files = expected_files(api, repo, rev)
    total, todo = sum(files.values()), missing_bytes(dest, files)
    if todo == 0:
        print(f"[{name}] already done: {dest} ({len(files)} files, {total / GB:.1f} GB)")
        return
    free = shutil.disk_usage(ROOT).free
    if free < todo + GB:
        raise RuntimeError(f"not enough disk space: need {todo / GB:.1f} GB, free {free / GB:.1f} GB")
    print(f"[{name}] {repo}@{rev[:8]} -> {dest}  ({todo / GB:.1f} of {total / GB:.1f} GB to download)")
    snapshot_download(repo, revision=rev, local_dir=dest, ignore_patterns=IGNORE)
    bad = [n for n, size in files.items() if not (dest / n).is_file() or (dest / n).stat().st_size != size]
    if bad:
        raise RuntimeError(f"size mismatch after download: {', '.join(bad)}")
    print(f"[{name}] done: {dest} ({len(files)} files, {total / GB:.1f} GB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="+", required=True, choices=[*MODELS, "all"], metavar="NAME",
                    help=f"one or more of: {', '.join(MODELS)}, or all")
    args = ap.parse_args()
    names = list(MODELS) if "all" in args.model else list(dict.fromkeys(args.model))

    ROOT.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    failed = []
    for name in names:  # one failure does not stop the others
        try:
            download(api, name)
        except Exception as e:
            print(f"[{name}] FAILED: {e}")
            failed.append(name)
    if failed:
        sys.exit(f"Failed: {', '.join(failed)}. Re-run the same command to resume.")
    print(f"All done: {', '.join(names)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.")
        sys.exit(130)
