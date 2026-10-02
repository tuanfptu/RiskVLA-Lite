#!/usr/bin/env python3
"""Launch the Streamlit research dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app = Path(__file__).resolve().parents[1] / "src" / "riskvla" / "demo" / "app.py"
    raise SystemExit(
        subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])
    )


if __name__ == "__main__":
    main()
