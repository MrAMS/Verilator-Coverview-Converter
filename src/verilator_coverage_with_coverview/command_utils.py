from __future__ import annotations

import shutil
import subprocess
import sys
from typing import NoReturn

TRANSFORM_BASE_ARGS = ["info-process", "transform", "--normalize-paths", "--normalize-hit-counts"]


def log(message: str) -> None:
    print(message, flush=True)


def fail(message: str) -> NoReturn:
    print(f"[ERROR] {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def need_cmd(name: str) -> None:
    if shutil.which(name) is None:
        fail(f"Missing required command: {name}")


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True)
