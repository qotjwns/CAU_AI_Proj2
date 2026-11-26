#!/usr/bin/env python3
"""
    Download COCO train2014 to data/coco/
      annotations/instances_train2014.json   object labels (from annotations_trainval2014.zip, 253 MB)
      annotations/instances_val2014.json     object labels of val2014 (MIHBench images), same zip
      train2014/                            82,783 images (from train2014.zip, 13.5 GB)
    python scripts/download_coco_train2014.py  # annotations first, then images; zips are deleted after extracting
"""
import argparse
import http.client
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "coco"
HOSTS = [
    "https://s3.amazonaws.com/images.cocodataset.org",
    "http://images.cocodataset.org",
]
ANN_ZIP, ANN_BYTES = "annotations/annotations_trainval2014.zip", 252_872_794  
ANN_FILES = ["annotations/instances_train2014.json", "annotations/instances_val2014.json"]
IMG_ZIP, IMG_BYTES = "zips/train2014.zip", 13_510_573_713
NAME, N_IMAGES = "train2014", 82_783
CHUNK = 8 << 20
RETRIES = 20
GB = 1e9
TTY = sys.stdout.isatty()


def log_progress(msg, last=False):
    # On a terminal, redraw one line; in a log file (nohup), write normal lines.
    if TTY:
        print("\r" + msg.ljust(79), end="\n" if last else "", flush=True)
    else:
        print(msg, flush=True)


def fmt_time(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def count_jpgs(folder):
    with os.scandir(folder) as it:
        return sum(1 for e in it if e.is_file() and e.name.lower().endswith(".jpg"))


def check_space(need):
    free = shutil.disk_usage(ROOT).free
    if free < need:
        sys.exit(f"Not enough disk space in {ROOT}: need {need / GB:.1f} GB, free {free / GB:.1f} GB.")


def pick_url(rel, size):
    for host in HOSTS:
        url = f"{host}/{rel}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as r:
                remote = int(r.headers["Content-Length"])
        except (OSError, KeyError, ValueError, TypeError) as e:
            print(f"  skip {url}: {e}")
            continue
        if remote != size:
            sys.exit(f"{url} reports {remote:,} bytes, expected {size:,}. The file changed upstream; stopping.")
        return url
    sys.exit(f"Could not reach any COCO download URL for {rel}.")


def download(url, part, size):
    every = 1 if TTY else 30  # seconds between progress lines
    for attempt in range(1, RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        if have == size:
            return
        err = None
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 206:  # server ignored Range: start from zero
                    have = 0
                with open(part, "ab" if have else "wb") as f:
                    t0 = last = time.time()
                    got = 0
                    while True:
                        buf = r.read(CHUNK)
                        if not buf:
                            break
                        f.write(buf)
                        got += len(buf)
                        now = time.time()
                        if now - last >= every:
                            last = now
                            done, speed = have + got, got / (now - t0)
                            log_progress(f"  {done / GB:6.2f} / {size / GB:.2f} GB ({100 * done / size:5.1f}%)"
                                         f"  {speed / 1e6:6.1f} MB/s  ETA {fmt_time((size - done) / speed)}")
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500 and e.code != 416:
                sys.exit(f"HTTP {e.code} from {url}; stopping.")
            err = e
        except (OSError, http.client.HTTPException) as e:  # timeouts, resets, truncated responses
            err = e
        have = part.stat().st_size if part.exists() else 0
        if have == size:
            log_progress(f"  {size / GB:.2f} GB downloaded.", last=True)
            return
        wait = min(60, 5 * attempt)
        log_progress(f"  stopped at {have / GB:.2f} GB ({err or 'connection closed early'}); "
                     f"retry {attempt}/{RETRIES} in {wait}s", last=True)
        time.sleep(wait)
    sys.exit("Download kept failing. Re-run the same command later to resume.")


def fetch_zip(rel, size, dest, unpacked):
    """Download HOST/rel to dest (resumable via dest.part) unless a complete copy is already there."""
    if dest.exists():
        if dest.stat().st_size != size:
            sys.exit(f"{dest} is {dest.stat().st_size:,} bytes, expected {size:,}. Delete it and re-run.")
        check_space(unpacked + GB)
        print(f"Using existing {dest}")
        return
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    check_space((size - have) + unpacked + GB)  # rest of the zip + what gets extracted from it
    url = pick_url(rel, size)
    print(f"Downloading {url}\n         -> {dest}" + (f"  (resuming at {have / GB:.2f} GB)" if have else ""))
    download(url, part, size)
    part.rename(dest)


def get_annotations():
    targets = [ROOT / f for f in ANN_FILES]
    if all(t.exists() for t in targets):
        print(f"Already done: {', '.join(map(str, targets))}")
        return
    zip_path = ROOT / Path(ANN_ZIP).name
    fetch_zip(ANN_ZIP, ANN_BYTES, zip_path, unpacked=GB)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name, t in zip(ANN_FILES, targets):
                print(f"Extracting {t}")
                t.parent.mkdir(parents=True, exist_ok=True)
                tmp = t.with_name(t.name + ".part")
                with zf.open(name) as src, open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)  # zipfile checks the CRC-32 when the read reaches the end
                tmp.rename(t)  # only a complete file ever appears under its real name
    except zipfile.BadZipFile as e:
        sys.exit(f"{zip_path} is corrupt ({e}). Delete it and re-run to download it again.")
    zip_path.unlink()
    print(f"Deleted {zip_path}")


def extract_images(zip_path):
    tmp = ROOT / f".{NAME}.extracting"
    if tmp.exists():  # left over from an interrupted extraction
        shutil.rmtree(tmp)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.infolist() if m.filename.startswith(NAME + "/") and not m.is_dir()]
            n_jpg = sum(m.filename.lower().endswith(".jpg") for m in members)
            if n_jpg != N_IMAGES:
                sys.exit(f"{zip_path} lists {n_jpg:,} images, expected {N_IMAGES:,}.")
            for i, m in enumerate(members, 1):
                zf.extract(m, tmp)  # zipfile verifies each file's CRC-32 as it reads it
                if i % 2000 == 0 or i == len(members):
                    log_progress(f"  extracted {i:,} / {len(members):,}", last=i == len(members))
    except zipfile.BadZipFile as e:
        sys.exit(f"\n{zip_path} is corrupt ({e}). Delete it and re-run to download it again.")
    (tmp / NAME).rename(ROOT / NAME)  # only a fully extracted folder ever appears as train2014/
    shutil.rmtree(tmp)


def get_images():
    final = ROOT / NAME
    if final.exists():
        n = count_jpgs(final)
        if n != N_IMAGES:
            sys.exit(f"{final} exists but holds {n:,} .jpg files (expected {N_IMAGES:,}). Delete it and re-run.")
        print(f"Already done: {final} ({n:,} images).")
        return
    zip_path = ROOT / Path(IMG_ZIP).name
    fetch_zip(IMG_ZIP, IMG_BYTES, zip_path, unpacked=IMG_BYTES)
    print(f"Extracting to {final}")
    extract_images(zip_path)
    n = count_jpgs(final)
    if n != N_IMAGES:
        sys.exit(f"{final} holds {n:,} .jpg files after extracting, expected {N_IMAGES:,}.")
    zip_path.unlink()
    print(f"Deleted {zip_path}")
    print(f"Done: {final} ({n:,} images)")


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    get_annotations()
    get_images()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.")
        sys.exit(130)
