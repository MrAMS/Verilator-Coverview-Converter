#!/usr/bin/env python3
"""Convert Verilator coverage.dat into a Coverview-ready archive."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

TOGGLE_INDEXED_PATTERN = re.compile(r"\[(\d+)\]:(0->1|1->0)$")
TOGGLE_SCALAR_PATTERN = re.compile(r":(0->1|1->0)$")
TRANSFORM_BASE_ARGS = ["info-process", "transform", "--normalize-paths", "--normalize-hit-counts"]


def log(message: str) -> None:
    print(message, flush=True)


def need_cmd(name: str) -> None:
    if shutil.which(name) is None:
        print(f"[ERROR] Missing required command: {name}", file=sys.stderr, flush=True)
        sys.exit(1)


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True)


def parse_string_list_field(payload: dict[str, object], key: str) -> list[str]:
    """Read an optional string-list field from JSON object with strict type checks."""
    value = payload.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        print(
            f"[ERROR] JSON field '{key}' must be a string array",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
    return value


def load_args_from_json(json_path: Path) -> list[str]:
    """
    Convert JSON object into flat CLI args.

    Supported schema:
    {input_dats, dataset, dats_root, sf_alias, exclude_sf}
    """
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"[ERROR] JSON args file not found: {json_path}", file=sys.stderr, flush=True)
        raise SystemExit(1)
    except OSError as exc:
        print(f"[ERROR] Failed to read JSON args file {json_path}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid JSON in {json_path}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)

    if not isinstance(payload, dict):
        print(
            "[ERROR] JSON args must be an object with keys: "
            "input_dats, dataset, dats_root, sf_alias, exclude_sf",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)

    allowed_keys = {"input_dats", "dataset", "dats_root", "sf_alias", "exclude_sf"}
    unknown_keys = sorted(key for key in payload if key not in allowed_keys)
    if unknown_keys:
        print(
            "[ERROR] Unsupported key(s) in JSON args: " + ", ".join(unknown_keys),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)

    args: list[str] = []
    args.extend(parse_string_list_field(payload, "input_dats"))

    dataset = payload.get("dataset")
    if dataset is not None:
        if not isinstance(dataset, str):
            print("[ERROR] JSON field 'dataset' must be a string", file=sys.stderr, flush=True)
            raise SystemExit(1)
        args.extend(["--dataset", dataset])

    dats_root = payload.get("dats_root")
    if dats_root is not None:
        if not isinstance(dats_root, str):
            print("[ERROR] JSON field 'dats_root' must be a string", file=sys.stderr, flush=True)
            raise SystemExit(1)
        args.extend(["--dats-root", dats_root])

    for alias in parse_string_list_field(payload, "sf_alias"):
        args.extend(["--sf-alias", alias])
    for path in parse_string_list_field(payload, "exclude_sf"):
        args.extend(["--exclude-sf", path])

    return args


def preprocess_argv_with_json(argv: list[str]) -> list[str]:
    """Expand --args-json before normal argparse parsing."""
    if "-h" in argv or "--help" in argv:
        return argv

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--args-json", action="append", metavar="FILE")
    parsed, filtered = parser.parse_known_args(argv)

    json_paths = parsed.args_json or []
    if len(json_paths) > 1:
        print("[ERROR] --args-json can only be provided once", file=sys.stderr, flush=True)
        raise SystemExit(1)
    if not json_paths:
        return filtered

    json_args = load_args_from_json(Path(json_paths[0]))
    if any(item == "--args-json" or item.startswith("--args-json=") for item in json_args):
        print(
            "[ERROR] Nested --args-json is not supported inside JSON args file",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)

    # JSON args are applied first so direct CLI flags can override them.
    return json_args + filtered


def normalize_coverage_path(path: str) -> str:
    path = path.strip()
    if not path:
        return ""
    return os.path.normpath(path)


@lru_cache(maxsize=512)
def compile_wildcard_prefix_regex(pattern: str) -> re.Pattern[str]:
    """Compile '*' wildcard prefix pattern with an optional '/...' tail."""
    fragments: list[str] = []
    wildcard_idx = 0
    for char in pattern:
        if char == "*":
            fragments.append(f"(?P<w{wildcard_idx}>.*)")
            wildcard_idx += 1
        else:
            fragments.append(re.escape(char))
    return re.compile("^" + "".join(fragments) + "(?P<tail>/.*)?$")


def apply_wildcard_template(template: str, values: list[str]) -> str | None:
    """Replace '*' in template with captured wildcard values in order."""
    output: list[str] = []
    value_idx = 0
    for char in template:
        if char == "*":
            if value_idx >= len(values):
                return None
            output.append(values[value_idx])
            value_idx += 1
        else:
            output.append(char)
    return "".join(output)


def rewrite_path_by_wildcard_prefix(path: str, src: str, dst: str) -> str | None:
    """Rewrite path with wildcard-aware prefix mapping."""
    if "*" not in src:
        return None

    matched = compile_wildcard_prefix_regex(src).match(path)
    if matched is None:
        return None

    wildcard_values = [matched.group(f"w{idx}") for idx in range(src.count("*"))]
    rewritten_prefix = apply_wildcard_template(dst, wildcard_values)
    if rewritten_prefix is None:
        return None

    tail = matched.group("tail") or ""
    return normalize_coverage_path(rewritten_prefix + tail)


def rewrite_path_by_exact_prefix(path: str, src: str, dst: str) -> str | None:
    if path == src:
        return dst
    prefix = "/" if src == "/" else src + "/"
    if path.startswith(prefix):
        suffix = path[len(src) :]
        return dst + suffix
    return None


def rewrite_path_by_prefix(path: str, src: str, dst: str) -> str | None:
    """Rewrite path when it equals src or is under src/ prefix."""
    if not src:
        return None

    wildcard_rewritten = rewrite_path_by_wildcard_prefix(path, src, dst)
    if wildcard_rewritten is not None:
        return wildcard_rewritten

    return rewrite_path_by_exact_prefix(path, src, dst)


def matches_excluded_pattern(path: str, pattern: str) -> bool:
    """
    Match one SF path against an exclude pattern.

    Supports both exact string matches and '*' wildcard matches.
    """
    normalized_path = normalize_coverage_path(path)
    normalized_pattern = normalize_coverage_path(pattern)

    if "*" in normalized_pattern:
        return fnmatch.fnmatchcase(normalized_path, normalized_pattern)

    return normalized_path == normalized_pattern


def parse_sf_aliases(raw_aliases: list[str]) -> list[tuple[str, str]]:
    aliases: list[tuple[str, str]] = []
    for item in raw_aliases:
        if "=" not in item:
            print(
                f"[ERROR] Invalid --sf-alias '{item}', expected FROM=TO",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(1)
        src, dst = item.split("=", 1)
        src = normalize_coverage_path(src)
        dst = normalize_coverage_path(dst)
        if not src:
            print(
                f"[ERROR] Invalid --sf-alias '{item}', FROM cannot be empty",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(1)
        src_wildcards = src.count("*")
        dst_wildcards = dst.count("*")
        if src_wildcards != dst_wildcards:
            print(
                f"[ERROR] Invalid --sf-alias '{item}', wildcard count must match on both sides",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(1)
        aliases.append((src, dst))

    # Prefer more specific source patterns first (longer non-wildcard text wins).
    aliases.sort(key=lambda pair: len(pair[0].replace("*", "")), reverse=True)
    return aliases


def rewrite_sf_path(path: str, aliases: list[tuple[str, str]]) -> str:
    for src, dst in aliases:
        rewritten = rewrite_path_by_prefix(path, src, dst)
        if rewritten is not None:
            return rewritten
    return path


def apply_sf_aliases(info_file: Path, aliases: list[tuple[str, str]]) -> int:
    if not aliases:
        return 0

    changed = 0
    output: list[str] = []
    with info_file.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("SF:"):
                old_path = line[3:].strip()
                new_path = rewrite_sf_path(old_path, aliases)
                if new_path != old_path:
                    changed += 1
                output.append(f"SF:{new_path}\n")
            else:
                output.append(line)
    info_file.write_text("".join(output), encoding="utf-8")
    return changed


def parse_excluded_sf_paths(raw_paths: list[str]) -> list[str]:
    parsed_paths: list[str] = []
    for item in raw_paths:
        parsed = normalize_coverage_path(item)
        if not parsed:
            print(
                f"[ERROR] Invalid --exclude-sf '{item}', path cannot be empty",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(1)
        parsed_paths.append(parsed)

    # Keep input order while removing duplicates.
    return list(dict.fromkeys(parsed_paths))


def expand_excluded_sf_paths(base_paths: list[str], aliases: list[tuple[str, str]]) -> set[str]:
    """
    Expand excludes through alias graph (both directions, transitively).

    Example:
      A->B and C->B, exclude C/foo.sv
      => also exclude B/foo.sv and A/foo.sv.
    """
    expanded: set[str] = set()
    for base in base_paths:
        queue = [base]
        seen = {base}
        while queue:
            current = queue.pop()
            for src, dst in aliases:
                for left, right in ((src, dst), (dst, src)):
                    rewritten = rewrite_path_by_prefix(current, left, right)
                    if rewritten is None or rewritten in seen:
                        continue
                    seen.add(rewritten)
                    queue.append(rewritten)
        expanded.update(seen)
    return expanded


def drop_excluded_sf_records(info_file: Path, excluded_paths: set[str]) -> tuple[int, int]:
    """Remove whole LCOV records whose SF matches excluded_paths."""
    if not excluded_paths:
        return 0, 0

    patterns = sorted(excluded_paths)
    kept: list[str] = []
    record: list[str] = []
    current_sf = ""
    removed_records = 0
    removed_files: set[str] = set()

    with info_file.open("r", encoding="utf-8") as file:
        for line in file:
            record.append(line)
            if line.startswith("SF:"):
                current_sf = normalize_coverage_path(line[3:])

            if not line.startswith("end_of_record"):
                continue

            if (
                current_sf
                and any(
                    matches_excluded_pattern(current_sf, pattern) for pattern in patterns
                )
            ):
                removed_records += 1
                removed_files.add(current_sf)
            else:
                kept.extend(record)

            record = []
            current_sf = ""

    if record:
        if (
            current_sf
            and any(matches_excluded_pattern(current_sf, pattern) for pattern in patterns)
        ):
            removed_records += 1
            removed_files.add(current_sf)
        else:
            kept.extend(record)

    info_file.write_text("".join(kept), encoding="utf-8")
    return removed_records, len(removed_files)


def squash_duplicate_sf_headers(info_file: Path) -> tuple[int, int]:
    """
    Keep only the first SF header inside a record.

    `info-process transform` can merge aliased records but leave repeated SF lines.
    Coverview expects one SF per record.
    """
    dropped = 0
    mismatched = 0
    output: list[str] = []
    current_sf: str | None = None

    with info_file.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("SF:"):
                sf = line[3:].strip()
                if current_sf is None:
                    current_sf = sf
                    output.append(f"SF:{sf}\n")
                else:
                    dropped += 1
                    if sf != current_sf:
                        mismatched += 1
                continue

            output.append(line)
            if line.startswith("end_of_record"):
                current_sf = None

    info_file.write_text("".join(output), encoding="utf-8")
    return dropped, mismatched


def export_coverage_info(input_dats: list[Path], output_info: Path, filter_type: str | None) -> None:
    """Export .dat files to LCOV .info with optional Verilator filter."""
    args = ["verilator_coverage"]
    if filter_type is not None:
        args.extend(["--filter-type", filter_type])
    args.extend(["-write-info", str(output_info), *(str(path) for path in input_dats)])
    run_cmd(args)


def collect_sf_paths(info_file: Path) -> list[str]:
    """Collect normalized SF paths from an LCOV info file."""
    sf_paths: list[str] = []
    with info_file.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("SF:"):
                path = normalize_coverage_path(line[3:])
                if path:
                    sf_paths.append(path)
    return sf_paths


def has_sf_records(info_file: Path) -> bool:
    with info_file.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("SF:"):
                return True
    return False


def compute_path_rewrite(info_files: list[Path]) -> tuple[str, str]:
    """
    Compute --strip-file-prefix regex + sources root from all SF paths.

    Returns empty strings when no safe common prefix exists.
    """
    sf_paths: list[str] = []
    for info_file in info_files:
        sf_paths.extend(collect_sf_paths(info_file))

    if not sf_paths:
        return "", ""

    is_abs = [os.path.isabs(path) for path in sf_paths]
    if any(is_abs) and not all(is_abs):
        # Mixed absolute/relative SF entries are ambiguous, keep original paths.
        return "", ""

    prefix = os.path.commonpath(sf_paths)
    if prefix in ("", "/"):
        return "", ""

    prefix = prefix.rstrip("/")
    strip_regex = re.escape(prefix) + "/?"
    sources_root = prefix if os.path.isabs(prefix) else os.path.abspath(prefix)
    return strip_regex, sources_root


def compact_toggle_comment(comment: str) -> str:
    """Keep only bit index + direction, e.g. '[3]:0->1'."""
    token = comment.strip()
    match = TOGGLE_INDEXED_PATTERN.search(token)
    if match is not None:
        return f"[{match.group(1)}]:{match.group(2)}"

    # Scalar (1-bit) points from Verilator are often emitted as `name:0->1`.
    # Treat them as bit 0 so we never produce ambiguous placeholders.
    direction = TOGGLE_SCALAR_PATTERN.search(token)
    if direction is not None:
        return f"[0]:{direction.group(1)}"

    # Keep an explicit fallback token to avoid empty BRDA names on malformed input.
    return "toggle"


def load_toggle_name_map_from_single_dat(dat_path: Path) -> dict[tuple[str, int], list[str]]:
    """Build (source-file, line) -> ordered compact toggle labels from one .dat file."""
    pattern = re.compile(r"C '(.*)'\s+(-?\d+)$")
    mapping: dict[tuple[str, int], list[str]] = defaultdict(list)

    for raw in dat_path.read_bytes().splitlines():
        if not raw.startswith(b"C '"):
            continue

        text = raw.decode("latin1")
        if "\x01t\x02toggle" not in text:
            continue

        matched = pattern.match(text)
        if matched is None:
            continue

        body = matched.group(1)
        fields: dict[str, str] = {}
        for part in body.split("\x01"):
            if "\x02" not in part:
                continue
            key, value = part.split("\x02", 1)
            fields[key] = value

        sf = fields.get("f")
        line = fields.get("l")
        comment = fields.get("o")
        if sf is None or line is None or comment is None:
            continue

        try:
            line_no = int(line)
        except ValueError:
            continue

        mapping[(sf, line_no)].append(compact_toggle_comment(comment))

    return mapping


def load_toggle_name_map(input_dats: list[Path]) -> tuple[dict[tuple[str, int], list[str]], int]:
    """Build stable toggle label mapping across one or more .dat files."""
    mapping: dict[tuple[str, int], list[str]] = {}
    conflicts = 0

    for dat_path in input_dats:
        current_map = load_toggle_name_map_from_single_dat(dat_path)
        for key, labels in current_map.items():
            if key not in mapping:
                mapping[key] = labels
            elif mapping[key] != labels:
                conflicts += 1

    return mapping, conflicts


def rewrite_toggle_brda_names(
    info_file: Path, mapping: dict[tuple[str, int], list[str]]
) -> tuple[int, int]:
    """Rewrite numeric BRDA names to descriptive toggle names from coverage.dat."""
    replaced = 0
    unresolved = 0
    current_sf = ""
    rewritten: list[str] = []
    with info_file.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("SF:"):
                current_sf = line[3:].strip()
                rewritten.append(line)
                continue

            if not line.startswith("BRDA:"):
                rewritten.append(line)
                continue

            params = line[5:].strip()
            parts = params.split(",", 3)
            if len(parts) != 4:
                rewritten.append(line)
                continue

            line_no, block_id, name, hit = parts
            target = name
            if current_sf and name.isdigit():
                try:
                    key = (current_sf, int(line_no))
                    index = int(name)
                except ValueError:
                    key = None
                    index = -1

                if key is not None:
                    labels = mapping.get(key)
                    if labels is not None and index < len(labels):
                        target = labels[index]
                    else:
                        unresolved += 1

            if target != name:
                replaced += 1
            rewritten.append(f"BRDA:{line_no},{block_id},{target},{hit}\n")

    info_file.write_text("".join(rewritten), encoding="utf-8")
    return replaced, unresolved


def ensure_lf_lh_summary(info_file: Path) -> None:
    """Guarantee LF/LH fields so Coverview can compute line coverage ratio."""
    output: list[str] = []
    line_hits: dict[int, int] = {}

    with info_file.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("SF:"):
                line_hits = {}
                output.append(line)
                continue

            if line.startswith("DA:"):
                output.append(line)
                payload = line[3:].strip()
                parts = payload.split(",", 1)
                if len(parts) == 2:
                    try:
                        line_no = int(parts[0])
                        hits = int(parts[1])
                    except ValueError:
                        continue
                    line_hits[line_no] = line_hits.get(line_no, 0) + hits
                continue

            if line.startswith("LF:") or line.startswith("LH:"):
                # Recompute summaries from DA entries for consistency.
                continue

            if line.startswith("end_of_record"):
                lf = len(line_hits)
                lh = sum(1 for hits in line_hits.values() if hits > 0)
                output.append(f"LF:{lf}\n")
                output.append(f"LH:{lh}\n")
                output.append(line)
                line_hits = {}
                continue

            output.append(line)

    info_file.write_text("".join(output), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = sys.argv[1:] if argv is None else argv
    effective_argv = preprocess_argv_with_json(raw_argv)

    parser = argparse.ArgumentParser(
        description="Convert one or more Verilator coverage.dat files into a Coverview input archive."
    )
    parser.add_argument(
        "--args-json",
        default=None,
        metavar="FILE",
        help=(
            "Load arguments from JSON object file with keys "
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
        print(
            "[ERROR] Input file(s) not found: " + ", ".join(str(path) for path in missing),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)

    return input_dats, dataset


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


def transform_info_file(info_file: Path, strip_regex: str, set_block_ids: bool = False) -> None:
    """Run common info-process transform options on one coverage file."""
    args = TRANSFORM_BASE_ARGS.copy()
    if set_block_ids:
        args.append("--set-block-ids")
    if strip_regex:
        args.extend(["--strip-file-prefix", strip_regex])
    run_cmd(args + [str(info_file)])


def apply_alias_and_exclusions(
    info_file: Path,
    coverage_label: str,
    sf_aliases: list[tuple[str, str]],
    excluded_sf_paths: set[str],
    set_block_ids: bool = False,
) -> None:
    """Apply SF alias merge + SF exclusion to one normalized coverage file."""
    if sf_aliases:
        alias_hits = apply_sf_aliases(info_file, sf_aliases)
        if alias_hits:
            # Re-run transform after alias rewrite so info-process can merge records.
            transform_info_file(info_file, strip_regex="", set_block_ids=set_block_ids)
            dropped, mismatched = squash_duplicate_sf_headers(info_file)
            log(
                f"      applied SF aliases on {coverage_label} info (rewritten SF: {alias_hits}, "
                f"dropped extra SF: {dropped}, mismatched: {mismatched})"
            )
        else:
            log(f"      SF aliases configured for {coverage_label} info but no path matched")

    if excluded_sf_paths:
        removed_records, removed_files = drop_excluded_sf_records(info_file, excluded_sf_paths)
        log(
            f"      excluded {coverage_label} coverage records: {removed_records} "
            f"(matched SF files: {removed_files})"
        )


def main() -> int:
    args = parse_args()

    # Parse alias/exclude intent once, then reuse across line/toggle/user stages.
    sf_aliases = parse_sf_aliases(args.sf_alias)
    excluded_sf_paths = expand_excluded_sf_paths(
        parse_excluded_sf_paths(args.exclude_sf), sf_aliases
    )
    if excluded_sf_paths:
        log(f"[0/9] Exclude SF paths: {len(excluded_sf_paths)} (after alias expansion)")
        for path in sorted(excluded_sf_paths):
            log(f"      - {path}")

    input_dats, dataset = resolve_inputs_and_dataset(args)
    outputs = build_output_files(dataset)

    need_cmd("verilator_coverage")
    need_cmd("info-process")

    # Stage 1: export raw LCOV data from Verilator.
    log(
        f"[1/9] Export combined coverage from {len(input_dats)} dat file(s) -> "
        f"{outputs.raw_all_info}"
    )
    export_coverage_info(input_dats, outputs.raw_all_info, filter_type=None)

    log(
        f"[2/9] Extract toggle coverage from {len(input_dats)} dat file(s) -> "
        f"{outputs.raw_toggle_info}"
    )
    export_coverage_info(input_dats, outputs.raw_toggle_info, filter_type="toggle")

    log(
        f"[3/9] Extract user coverage from {len(input_dats)} dat file(s) -> "
        f"{outputs.raw_user_info}"
    )
    export_coverage_info(input_dats, outputs.raw_user_info, filter_type="user")

    # Stage 2: determine whether user coverage exists to avoid packing empty files.
    has_user_coverage = has_sf_records(outputs.raw_user_info)
    if has_user_coverage:
        log(f"      detected user coverage records, will include {outputs.user_info}")
    else:
        log("      no user coverage records in this run")
        if outputs.user_info.exists():
            outputs.user_info.unlink()

    raw_info_files = [outputs.raw_all_info, outputs.raw_toggle_info]
    if has_user_coverage:
        raw_info_files.append(outputs.raw_user_info)

    log("[4/9] Analyze SF paths for prefix stripping")
    strip_regex, sources_root = compute_path_rewrite(raw_info_files)
    if strip_regex:
        log(f"      strip regex: {strip_regex}")
        log(f"      sources root: {sources_root}")
    else:
        log("      no stable common SF prefix found; keep original SF paths")

    # Stage 3: line coverage pipeline.
    run_cmd(
        [
            "info-process",
            "extract",
            "--coverage-type",
            "line",
            "--output",
            str(outputs.line_info),
            str(outputs.raw_all_info),
        ]
    )
    log(f"[5/9] Normalize line coverage -> {outputs.line_info}")
    transform_info_file(outputs.line_info, strip_regex=strip_regex)
    apply_alias_and_exclusions(
        outputs.line_info,
        coverage_label="line",
        sf_aliases=sf_aliases,
        excluded_sf_paths=excluded_sf_paths,
    )
    ensure_lf_lh_summary(outputs.line_info)

    # Stage 4: toggle coverage pipeline (with BRDA label rewrite).
    shutil.copyfile(outputs.raw_toggle_info, outputs.toggle_info)
    toggle_name_map, mapping_conflicts = load_toggle_name_map(input_dats)
    renamed, unresolved = rewrite_toggle_brda_names(outputs.toggle_info, toggle_name_map)
    log(
        f"[6/9] Canonicalize toggle BRDA names in {outputs.toggle_info} "
        f"(rewritten: {renamed}, unresolved: {unresolved}, conflicts: {mapping_conflicts})"
    )
    log(f"[7/9] Normalize toggle coverage -> {outputs.toggle_info}")
    transform_info_file(outputs.toggle_info, strip_regex=strip_regex, set_block_ids=True)
    apply_alias_and_exclusions(
        outputs.toggle_info,
        coverage_label="toggle",
        sf_aliases=sf_aliases,
        excluded_sf_paths=excluded_sf_paths,
        set_block_ids=True,
    )

    # Stage 5: optional user coverage pipeline.
    coverage_files = [outputs.line_info, outputs.toggle_info]
    if has_user_coverage:
        shutil.copyfile(outputs.raw_user_info, outputs.user_info)
        log(f"[8/9] Normalize user coverage -> {outputs.user_info}")
        transform_info_file(outputs.user_info, strip_regex=strip_regex)
        apply_alias_and_exclusions(
            outputs.user_info,
            coverage_label="user",
            sf_aliases=sf_aliases,
            excluded_sf_paths=excluded_sf_paths,
        )
        coverage_files.append(outputs.user_info)
    else:
        log("[8/9] Skip user coverage normalization (no user records)")

    # Stage 6: pack Coverview archive.
    outputs.config_json.write_text("{}\n", encoding="utf-8")

    pack_args = [
        "info-process",
        "pack",
        "--output",
        str(outputs.output_zip),
        "--config",
        str(outputs.config_json),
        "--coverage-files",
        *(str(path) for path in coverage_files),
        "--generate-tables",
        "line",
    ]
    if sources_root:
        pack_args.extend(["--sources-root", sources_root])

    log(f"[9/9] Pack Coverview archive -> {outputs.output_zip}")
    run_cmd(pack_args)

    log(f"[OK] Done. Import {outputs.output_zip} into Coverview.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
