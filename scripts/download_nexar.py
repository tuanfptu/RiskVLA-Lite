#!/usr/bin/env python3
"""Download the official Nexar collision-prediction dataset.

Example, on the persistent GPU machine:

    python scripts/download_nexar.py --output /path/to/nexar

The script prints the published disk requirement before any transfer, resumes
incomplete files, and never prints HF_TOKEN. This development environment
should use --dry-run rather than starting the full download.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from riskvla.data.nexar import DATASET_REVISION
from riskvla.data.nexar_download import (
    SPLIT_ALLOW_PATTERNS,
    NexarDownloadError,
    download_nexar,
    read_hf_token,
    sanitize_secret,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--split",
        choices=tuple(SPLIT_ALLOW_PATTERNS),
        default="all",
    )
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the disk estimate and exit without contacting Hugging Face.",
    )
    args = parser.parse_args(argv)
    token = None if args.dry_run else read_hf_token()
    try:
        download_nexar(
            args.output,
            split=args.split,
            revision=args.revision,
            token=token,
            dry_run=args.dry_run,
        )
    except NexarDownloadError as exc:
        print(sanitize_secret(str(exc), token), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
