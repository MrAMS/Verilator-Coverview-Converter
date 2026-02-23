from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputFiles:
    raw_all_info: Path
    raw_toggle_info: Path
    raw_user_info: Path
    line_info: Path
    toggle_info: Path
    user_info: Path
    config_json: Path
    output_zip: Path


def build_output_files(dataset: str) -> OutputFiles:
    return OutputFiles(
        raw_all_info=Path(f"coverage_all_raw_{dataset}.info"),
        raw_toggle_info=Path(f"coverage_toggle_raw_{dataset}.info"),
        raw_user_info=Path(f"coverage_user_raw_{dataset}.info"),
        line_info=Path(f"coverage_line_{dataset}.info"),
        toggle_info=Path(f"coverage_toggle_{dataset}.info"),
        user_info=Path(f"coverage_user_{dataset}.info"),
        config_json=Path(f"coverview_config_{dataset}.json"),
        output_zip=Path(f"coverview_data_{dataset}.zip"),
    )
