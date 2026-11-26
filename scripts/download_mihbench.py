#!/usr/bin/env python3
"""
    Download MIHBench
      questions/<task>.jsonl     
      images/coco_val2014/       
      images/co3d/               

    python scripts/download_mihbench.py  # safe to re-run; finished files are skipped
"""
import argparse
import hashlib
import http.client
import json
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from download_coco_train2014 import download  # resumable download with retries and progress

ROOT = Path(__file__).resolve().parents[1] / "data" / "mihbench"
QUESTION_DIR = ROOT / "questions"
COCO_DIR = ROOT / "images" / "coco_val2014"
CO3D_DIR = ROOT / "images" / "co3d"

COMMIT = "e834d4691357b77c9a86e89cc90a5c79873546d4"  # pgtrece/DAB main on 2026-11-26
QUESTIONS_URL = f"https://raw.githubusercontent.com/pgtrece/DAB/{COMMIT}/Questions"
QUESTIONS = {  # local task name -> path in the repo
    "count": "Count/comparison_same_2.jsonl",
    "existence_adversarial": "Existence/new_adv/coco_pope_adversarial_3.jsonl",
    "existence_popular": "Existence/new_pop/coco_pope_popular_3.jsonl",
    "existence_random": "Existence/new_ran/coco_pope_random_3.jsonl",
    "id_consistency": "Id_Consitency/clip_questions_most_different_4.jsonl",  # "Consitency" is the repo's spelling
}
N_QUESTIONS = 800

COCO_TASKS, N_COCO_IMAGES = ["count", "existence_adversarial", "existence_popular", "existence_random"], 1_908
COCO_NAME = re.compile(r"COCO_val2014_\d{12}\.jpg")  # names come from downloaded files; never trust them as paths
COCO_HOSTS = [  # the same S3 bucket as download_coco_train2014.py
    "https://s3.amazonaws.com/images.cocodataset.org/val2014",
    "http://images.cocodataset.org/val2014",
]
WORKERS = 16
RETRIES = 5

CO3D_TASKS, N_CO3D_IMAGES = ["id_consistency"], 1_619
CO3D_NAME = re.compile(r"[a-z]+/\d+_\d+_\d+/frame\d{6}\.jpg")  # category/sequence/frame
CO3D_FOLDER = "https://drive.google.com/drive/folders/1aiI109w3wXUDUxPOAdd_eY-WKnkgK6Vg"  # linked from the DAB README
CO3D_URL = "https://drive.usercontent.google.com/download?id=1rUSdsXDw9R_dkPVs-v8S2BXIhvxKM13G&export=download&confirm=t"
CO3D_TAR, CO3D_BYTES = ROOT / "CO3D.tar.gz", 1_724_037_005
CO3D_SHA256 = "f911c8af3aa0d9c23bc1141aeee32325d4d96f69685b6aef8aca7aff42124e01"
CO3D_PREFIX = "mnt/data/wmr/wmr_tmp/ljl/datasets/CO3D/"  # the authors' own path, stored inside the tar


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()  # raises IncompleteRead if the body is shorter than Content-Length
        length = r.headers.get("Content-Length")
        if length is not None and len(data) != int(length):
            raise OSError(f"got {len(data):,} bytes, Content-Length says {int(length):,}")
        return data, r.headers.get("ETag", "").strip('"')


def write_atomic(dest, data):
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    tmp.rename(dest)  # only a complete file ever appears under its real name


def read_questions(data, task):
    rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    if len(rows) != N_QUESTIONS:
        sys.exit(f"{task}: {len(rows)} questions, expected {N_QUESTIONS}.")
    return rows


def get_questions():
    QUESTION_DIR.mkdir(parents=True, exist_ok=True)
    for task, rel in QUESTIONS.items():
        dest = QUESTION_DIR / f"{task}.jsonl"
        if dest.exists():
            print(f"Already done: {dest}")
            continue
        try:
            data, _ = fetch(f"{QUESTIONS_URL}/{rel}")
        except (OSError, http.client.HTTPException) as e:
            sys.exit(f"Could not download {QUESTIONS_URL}/{rel}: {e}")
        read_questions(data, task)
        write_atomic(dest, data)
        print(f"Saved {dest}")


def image_names(tasks, pattern, expected):
    names = set()
    for task in tasks:
        for row in read_questions((QUESTION_DIR / f"{task}.jsonl").read_bytes(), task):
            names.update(row["image_list"])
    bad = sorted(n for n in names if not pattern.fullmatch(n))
    if bad:
        sys.exit(f"Unexpected image names in {', '.join(tasks)}, e.g. {bad[:3]}; stopping.")
    if len(names) != expected:
        sys.exit(f"{', '.join(tasks)} use {len(names):,} images, expected {expected:,}.")
    return sorted(names)


def download_coco_image(name):
    err = None
    for attempt in range(1, RETRIES + 1):
        for host in COCO_HOSTS:
            try:
                data, etag = fetch(f"{host}/{name}")
                if re.fullmatch(r"[0-9a-f]{32}", etag) and hashlib.md5(data).hexdigest() != etag:
                    raise OSError("MD5 does not match the server's ETag")  # S3's ETag is the MD5 for plain uploads
                if not data.startswith(b"\xff\xd8"):
                    raise OSError("not a JPEG")
                write_atomic(COCO_DIR / name, data)
                return None
            except (OSError, http.client.HTTPException) as e:  # HTTPError, timeouts, resets, bad content
                err = e
        time.sleep(2 * attempt)
    return f"{name}: {err}"


def get_coco(names):
    COCO_DIR.mkdir(parents=True, exist_ok=True)
    for leftover in COCO_DIR.glob("*.part"):  # from an interrupted run
        leftover.unlink()
    todo = [n for n in names if not (COCO_DIR / n).exists()]
    if not todo:
        print(f"Already done: {COCO_DIR} ({len(names):,} images).")
        return
    print(f"Downloading {len(todo):,} of {len(names):,} images -> {COCO_DIR}")
    failed = []
    with ThreadPoolExecutor(WORKERS) as pool:
        for i, fut in enumerate(as_completed(pool.submit(download_coco_image, n) for n in todo), 1):
            if (msg := fut.result()) is not None:
                failed.append(msg)
            if i % 200 == 0 or i == len(todo):
                print(f"  {i:,} / {len(todo):,}" + (f"  ({len(failed)} failed)" if failed else ""), flush=True)
    if failed:
        print("\n".join(failed[:10]))
        sys.exit(f"{len(failed)} images failed. Re-run the same command to retry them.")
    print(f"Done: {COCO_DIR} ({len(names):,} images)")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while buf := f.read(8 << 20):
            h.update(buf)
    return h.hexdigest()


def fetch_co3d_tar():
    if CO3D_TAR.exists():
        print(f"Using existing {CO3D_TAR}")
        return
    # Google Drive answers with an HTML page instead of the file when its download quota is used up.
    try:
        req = urllib.request.Request(CO3D_URL, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            ctype, total = r.headers.get("Content-Type", ""), r.headers.get("Content-Range", "").rpartition("/")[2]
    except (OSError, http.client.HTTPException) as e:
        ctype, total = str(e), ""
    if ctype.startswith("text/html") or total != str(CO3D_BYTES):
        sys.exit(f"Google Drive did not serve CO3D.tar.gz ({ctype}, {total or '?'} bytes). Try again later, "
                 f"or download it by hand from {CO3D_FOLDER} to {CO3D_TAR} and re-run.")
    part = CO3D_TAR.with_name(CO3D_TAR.name + ".part")
    print(f"Downloading CO3D.tar.gz from Google Drive\n         -> {CO3D_TAR}")
    download(CO3D_URL, part, CO3D_BYTES)
    part.rename(CO3D_TAR)


def get_co3d(names):
    todo = {n for n in names if not (CO3D_DIR / n).exists()}
    if not todo:
        print(f"Already done: {CO3D_DIR} ({len(names):,} images).")
        return
    fetch_co3d_tar()
    print(f"Checking {CO3D_TAR}")
    if (digest := sha256(CO3D_TAR)) != CO3D_SHA256:
        sys.exit(f"{CO3D_TAR} has sha256 {digest}, expected {CO3D_SHA256}. Delete it and re-run.")
    print(f"Extracting {len(todo):,} of the frames in it -> {CO3D_DIR}")
    with tarfile.open(CO3D_TAR, "r|gz") as tf:
        for m in tf:  # the tar holds more frames than the questions use; take only those
            rel = m.name.removeprefix(CO3D_PREFIX)
            if m.isfile() and rel in todo:
                dest = CO3D_DIR / rel  # rel is one of our validated names, never a raw tar path
                dest.parent.mkdir(parents=True, exist_ok=True)
                write_atomic(dest, tf.extractfile(m).read())
                todo.discard(rel)
    if todo:
        sys.exit(f"{CO3D_TAR} lacks {len(todo)} frames the questions use, e.g. {sorted(todo)[:3]}.")
    CO3D_TAR.unlink()
    print(f"Deleted {CO3D_TAR}")
    print(f"Done: {CO3D_DIR} ({len(names):,} images)")


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    get_questions()
    get_coco(image_names(COCO_TASKS, COCO_NAME, N_COCO_IMAGES))
    get_co3d(image_names(CO3D_TASKS, CO3D_NAME, N_CO3D_IMAGES))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.")
        sys.exit(130)
