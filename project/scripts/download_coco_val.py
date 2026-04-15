#!/usr/bin/env python3
"""
Download COCO val2017 images and train/val annotations from the official site
(no Google Drive / OAuth). Extracts under data/coco/ to match config.py paths.
"""
from __future__ import annotations

import argparse
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

# Official COCO mirrors (see https://cocodataset.org/#download).
# Default is HTTP — same files as on the site; avoids TLS issues (proxies / cert mismatch on HTTPS).
VAL_ZIP_URL_HTTP = "http://images.cocodataset.org/zips/val2017.zip"
ANN_ZIP_URL_HTTP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
VAL_ZIP_URL_HTTPS = "https://images.cocodataset.org/zips/val2017.zip"
ANN_ZIP_URL_HTTPS = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"

CHUNK = 1024 * 1024  # 1 MiB


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def coco_root() -> Path:
    return project_root() / "data" / "coco"


def download(url: str, dest: Path, *, ssl_context: ssl.SSLContext | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading:\n  {url}\n  -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "coco-download/1.0"})
    open_kw: dict = {}
    if url.startswith("https://") and ssl_context is not None:
        open_kw["context"] = ssl_context
    try:
        with urllib.request.urlopen(req, **open_kw) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            read = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    read += len(chunk)
                    if total and read % (50 * CHUNK) < CHUNK:
                        pct = 100.0 * read / total
                        print(f"  {pct:5.1f}% ({read // (1024 * 1024)} MiB)", flush=True)
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def extract_zip(zip_path: Path, into: Path) -> None:
    into.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} -> {into}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(into)


def have_val_images(root: Path) -> bool:
    val_dir = root / "val2017"
    if not val_dir.is_dir():
        return False
    # 5000 images expected; avoid counting all — check a few jpgs
    return any(val_dir.glob("*.jpg"))


def have_instances_val(root: Path) -> bool:
    p = root / "annotations" / "instances_val2017.json"
    return p.is_file() and p.stat().st_size > 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if data already exists.",
    )
    parser.add_argument(
        "--https",
        action="store_true",
        help="Use https:// URLs instead of http:// (default: plain HTTP, per cocodataset.org).",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="With --https, do not verify TLS certificates (last resort behind broken proxies).",
    )
    args = parser.parse_args()

    val_url = VAL_ZIP_URL_HTTPS if args.https else VAL_ZIP_URL_HTTP
    ann_url = ANN_ZIP_URL_HTTPS if args.https else ANN_ZIP_URL_HTTP
    ssl_ctx: ssl.SSLContext | None = None
    if args.https:
        ssl_ctx = ssl._create_unverified_context() if args.insecure else ssl.create_default_context()

    root = coco_root()
    root.mkdir(parents=True, exist_ok=True)
    cache = root / ".download_cache"
    cache.mkdir(parents=True, exist_ok=True)

    val_zip = cache / "val2017.zip"
    ann_zip = cache / "annotations_trainval2017.zip"

    if not args.force and have_val_images(root) and have_instances_val(root):
        print(f"COCO val2017 already present under {root}; use --force to replace.")
        return 0

    if args.force:
        import shutil

        for p in (root / "val2017", root / "annotations"):
            if p.exists():
                shutil.rmtree(p)
        for z in (val_zip, ann_zip):
            z.unlink(missing_ok=True)

    if not val_zip.is_file():
        download(val_url, val_zip, ssl_context=ssl_ctx)
    else:
        print(f"Using existing {val_zip}")

    if not ann_zip.is_file():
        download(ann_url, ann_zip, ssl_context=ssl_ctx)
    else:
        print(f"Using existing {ann_zip}")

    extract_zip(val_zip, root)
    extract_zip(ann_zip, root)

    print("Done. Expected paths:")
    print(f"  {root / 'val2017'}")
    print(f"  {root / 'annotations' / 'instances_val2017.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
