from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from .command_utils import fail

try:
    import yaml
except ModuleNotFoundError:
    fail("Missing required Python package: pyyaml. Install dependencies with: uv sync")


@dataclass(frozen=True)
class YamlArgs:
    input_dats: list[str]
    dataset: str | None
    dats_root: str | None
    sf_alias: dict[str, list[str]]
    exclude_sf: list[str]


def parse_string_list_field(payload: dict[str, object], key: str) -> list[str]:
    """Read an optional string-list field from YAML object with strict type checks."""
    value = payload.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        fail(f"YAML field '{key}' must be a string array")
    return value


def parse_string_list_map_field(payload: dict[str, object], key: str) -> dict[str, list[str]]:
    """
    Read an optional string->string-list mapping from YAML object.

    Used for grouped sf_alias declarations.
    """
    value = payload.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        fail(f"YAML field '{key}' must be a mapping of group name to string array")

    parsed: dict[str, list[str]] = {}
    for raw_group, raw_items in value.items():
        if not isinstance(raw_group, str) or not raw_group.strip():
            fail(f"YAML field '{key}' group name must be a non-empty string")
        if not isinstance(raw_items, list) or any(not isinstance(item, str) for item in raw_items):
            fail(
                f"YAML field '{key}.{raw_group}' must be a string array "
                "(first item is canonical path, others are paths or group refs)"
            )
        parsed[raw_group.strip()] = raw_items[:]
    return parsed


def load_yaml(yaml_path: Path) -> object:
    """Load args payload from YAML text."""
    try:
        raw_text = yaml_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        fail(f"YAML args file not found: {yaml_path}")
    except OSError as exc:
        fail(f"Failed to read YAML args file {yaml_path}: {exc}")

    try:
        return yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        fail(f"Invalid YAML in {yaml_path}: {exc}")


def load_args_from_yaml(yaml_path: Path) -> YamlArgs:
    """Load structured YAML args payload."""
    payload = load_yaml(yaml_path)

    if not isinstance(payload, dict):
        fail(
            "YAML args must be an object with keys: "
            "input_dats, dataset, dats_root, sf_alias, exclude_sf"
        )

    allowed_keys = {"input_dats", "dataset", "dats_root", "sf_alias", "exclude_sf"}
    unknown_keys = sorted(key for key in payload if key not in allowed_keys)
    if unknown_keys:
        fail("Unsupported key(s) in YAML args: " + ", ".join(unknown_keys))

    dataset = payload.get("dataset")
    if dataset is not None:
        if not isinstance(dataset, str):
            fail("YAML field 'dataset' must be a string")

    dats_root = payload.get("dats_root")
    if dats_root is not None:
        if not isinstance(dats_root, str):
            fail("YAML field 'dats_root' must be a string")

    return YamlArgs(
        input_dats=parse_string_list_field(payload, "input_dats"),
        dataset=dataset,
        dats_root=dats_root,
        sf_alias=parse_string_list_map_field(payload, "sf_alias"),
        exclude_sf=parse_string_list_field(payload, "exclude_sf"),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = sys.argv[1:] if argv is None else argv

    parser = argparse.ArgumentParser(
        prog="convert-coverage-to-coverview",
        description="Convert Verilator coverage.dat files into a Coverview input archive using YAML config.",
    )
    parser.add_argument(
        "--args-yaml",
        required=True,
        metavar="FILE",
        help=(
            "Load arguments from YAML object file with keys "
            "{input_dats,dataset,dats_root,sf_alias,exclude_sf}."
        ),
    )
    args = parser.parse_args(raw_argv)
    yaml_args = load_args_from_yaml(Path(args.args_yaml))

    return argparse.Namespace(
        input_dats=yaml_args.input_dats,
        dataset=yaml_args.dataset,
        dats_root=yaml_args.dats_root,
        sf_alias=yaml_args.sf_alias,
        exclude_sf=yaml_args.exclude_sf,
    )


def resolve_input_dat_path(raw_path: str, dats_root: Path | None) -> Path:
    path = Path(raw_path)
    if dats_root is not None and not path.is_absolute():
        return dats_root / path
    return path


def resolve_inputs_and_dataset(args: argparse.Namespace) -> tuple[list[Path], str]:
    raw_inputs: list[str] = args.input_dats[:] if args.input_dats else ["coverage.dat"]
    dataset = args.dataset
    dats_root = Path(args.dats_root).expanduser() if args.dats_root else None

    if dataset is None:
        dataset = "verilator"

    input_dats = [resolve_input_dat_path(path, dats_root) for path in raw_inputs]
    missing = [path for path in input_dats if not path.is_file()]
    if missing:
        fail("Input file(s) not found: " + ", ".join(str(path) for path in missing))

    return input_dats, dataset
