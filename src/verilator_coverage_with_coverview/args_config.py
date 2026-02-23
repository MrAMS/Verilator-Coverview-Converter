from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .command_utils import fail

try:
    import yaml
except ModuleNotFoundError:
    fail("Missing required Python package: pyyaml. Install dependencies with: uv sync")


def parse_string_list_field(payload: dict[str, object], key: str) -> list[str]:
    """Read an optional string-list field from YAML object with strict type checks."""
    value = payload.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        fail(f"YAML field '{key}' must be a string array")
    return value


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


def load_args_from_yaml(yaml_path: Path) -> list[str]:
    """
    Convert YAML object into flat CLI args.

    Supported schema:
    {input_dats, dataset, dats_root, sf_alias, exclude_sf}
    """
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

    args: list[str] = []
    args.extend(parse_string_list_field(payload, "input_dats"))

    dataset = payload.get("dataset")
    if dataset is not None:
        if not isinstance(dataset, str):
            fail("YAML field 'dataset' must be a string")
        args.extend(["--dataset", dataset])

    dats_root = payload.get("dats_root")
    if dats_root is not None:
        if not isinstance(dats_root, str):
            fail("YAML field 'dats_root' must be a string")
        args.extend(["--dats-root", dats_root])

    for alias in parse_string_list_field(payload, "sf_alias"):
        args.extend(["--sf-alias", alias])
    for path in parse_string_list_field(payload, "exclude_sf"):
        args.extend(["--exclude-sf", path])

    return args


def preprocess_argv_with_yaml(argv: list[str]) -> list[str]:
    """Expand --args-yaml before normal argparse parsing."""
    if "-h" in argv or "--help" in argv:
        return argv

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--args-yaml", action="append", metavar="FILE")
    parsed, filtered = parser.parse_known_args(argv)

    yaml_paths = parsed.args_yaml or []
    if len(yaml_paths) > 1:
        fail("--args-yaml can only be provided once")
    if not yaml_paths:
        return filtered

    yaml_args = load_args_from_yaml(Path(yaml_paths[0]))
    if any(item == "--args-yaml" or item.startswith("--args-yaml=") for item in yaml_args):
        fail("Nested --args-yaml is not supported inside YAML args file")

    # YAML args are applied first so direct CLI flags can override them.
    return yaml_args + filtered


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = sys.argv[1:] if argv is None else argv
    effective_argv = preprocess_argv_with_yaml(raw_argv)

    parser = argparse.ArgumentParser(
        prog="convert-coverage-to-coverview",
        description="Convert one or more Verilator coverage.dat files into a Coverview input archive.",
    )
    parser.add_argument(
        "--args-yaml",
        default=None,
        metavar="FILE",
        help=(
            "Load arguments from YAML object file with keys "
            "{input_dats,dataset,dats_root,sf_alias,exclude_sf}."
        ),
    )
    parser.add_argument(
        "input_dats",
        nargs="*",
        help="Input coverage.dat paths; pass multiple files to merge their coverage.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default=None,
        help="Dataset name for output files (default: verilator).",
    )
    parser.add_argument(
        "--dats-root",
        default=None,
        metavar="DIR",
        help=(
            "Optional common prefix for input dat paths. "
            "Relative input_dats are resolved under this directory."
        ),
    )
    parser.add_argument(
        "--sf-alias",
        action="append",
        default=[],
        metavar="FROM=TO",
        help=(
            "Map equivalent source paths (can be repeated). "
            "When SF is FROM or starts with FROM/, it will be rewritten to TO. "
            "Supports '*' wildcard (same count required on both sides)."
        ),
    )
    parser.add_argument(
        "--exclude-sf",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Exclude the specified SF path from output coverage (repeatable). "
            "Supports '*' wildcard. When combined with --sf-alias, "
            "equivalent aliased paths are also excluded."
        ),
    )
    return parser.parse_args(effective_argv)


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
