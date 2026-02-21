#!/usr/bin/env python3
"""Convert Verilator coverage.dat into a Coverview-ready archive."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

TOGGLE_INDEXED_PATTERN = re.compile(r"\[(\d+)\]:(0->1|1->0)$")
TOGGLE_SCALAR_PATTERN = re.compile(r":(0->1|1->0)$")


def log(message: str) -> None:
    print(message, flush=True)


def need_cmd(name: str) -> None:
    if shutil.which(name) is None:
        print(f"[ERROR] Missing required command: {name}", file=sys.stderr, flush=True)
        sys.exit(1)


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True)


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
        src = src.strip()
        dst = dst.strip()
        if not src:
            print(
                f"[ERROR] Invalid --sf-alias '{item}', FROM cannot be empty",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(1)
        aliases.append((src, dst))

    # Prefer the longest source prefix first, so specific rules win.
    aliases.sort(key=lambda pair: len(pair[0]), reverse=True)
    return aliases


def rewrite_sf_path(path: str, aliases: list[tuple[str, str]]) -> str:
    for src, dst in aliases:
        src_norm = src.rstrip("/")
        dst_norm = dst.rstrip("/")
        if path == src_norm:
            return dst_norm
        prefix = src_norm + "/"
        if path.startswith(prefix):
            suffix = path[len(src_norm) :]
            return dst_norm + suffix
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
    args = ["verilator_coverage"]
    if filter_type is not None:
        args.extend(["--filter-type", filter_type])
    args.extend(["-write-info", str(output_info), *(str(path) for path in input_dats)])
    run_cmd(args)


def collect_sf_paths(info_file: Path) -> list[str]:
    sf_paths: list[str] = []
    with info_file.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("SF:"):
                path = os.path.normpath(line[3:].strip())
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one or more Verilator coverage.dat files into a Coverview input archive."
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
        "--sf-alias",
        action="append",
        default=[],
        metavar="FROM=TO",
        help=(
            "Map equivalent source paths (can be repeated). "
            "When SF is FROM or starts with FROM/, it will be rewritten to TO."
        ),
    )
    return parser.parse_args()


def resolve_inputs_and_dataset(args: argparse.Namespace) -> tuple[list[Path], str]:
    raw_inputs: list[str] = args.input_dats[:] if args.input_dats else ["coverage.dat"]
    dataset = args.dataset

    # Backward compatibility: old form was "<input_dat> <dataset>".
    if dataset is None and len(raw_inputs) >= 2:
        maybe_dataset = raw_inputs[-1]
        maybe_path = Path(maybe_dataset)
        maybe_dataset_like = "/" not in maybe_dataset and not maybe_dataset.endswith(".dat")
        if maybe_dataset_like and not maybe_path.exists():
            prev_inputs = [Path(p) for p in raw_inputs[:-1]]
            if all(path.is_file() for path in prev_inputs):
                dataset = maybe_dataset
                raw_inputs = raw_inputs[:-1]

    if dataset is None:
        dataset = "verilator"

    input_dats = [Path(path) for path in raw_inputs]
    missing = [path for path in input_dats if not path.is_file()]
    if missing:
        print(
            "[ERROR] Input file(s) not found: " + ", ".join(str(path) for path in missing),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)

    return input_dats, dataset


def main() -> int:
    args = parse_args()
    sf_aliases = parse_sf_aliases(args.sf_alias)

    input_dats, dataset = resolve_inputs_and_dataset(args)
    raw_all_info = Path(f"coverage_all_raw_{dataset}.info")
    raw_toggle_info = Path(f"coverage_toggle_raw_{dataset}.info")
    raw_user_info = Path(f"coverage_user_raw_{dataset}.info")
    line_info = Path(f"coverage_line_{dataset}.info")
    toggle_info = Path(f"coverage_toggle_{dataset}.info")
    user_info = Path(f"coverage_user_{dataset}.info")
    config_json = Path(f"coverview_config_{dataset}.json")
    output_zip = Path(f"coverview_data_{dataset}.zip")

    need_cmd("verilator_coverage")
    need_cmd("info-process")

    log(f"[1/9] Export combined coverage from {len(input_dats)} dat file(s) -> {raw_all_info}")
    export_coverage_info(input_dats, raw_all_info, filter_type=None)

    log(f"[2/9] Extract toggle coverage from {len(input_dats)} dat file(s) -> {raw_toggle_info}")
    export_coverage_info(input_dats, raw_toggle_info, filter_type="toggle")

    log(f"[3/9] Extract user coverage from {len(input_dats)} dat file(s) -> {raw_user_info}")
    export_coverage_info(input_dats, raw_user_info, filter_type="user")
    has_user_coverage = has_sf_records(raw_user_info)
    if has_user_coverage:
        log(f"      detected user coverage records, will include {user_info}")
    else:
        log("      no user coverage records in this run")
        if user_info.exists():
            user_info.unlink()

    raw_info_files = [raw_all_info, raw_toggle_info]
    if has_user_coverage:
        raw_info_files.append(raw_user_info)

    log("[4/9] Analyze SF paths for prefix stripping")
    strip_regex, sources_root = compute_path_rewrite(raw_info_files)
    if strip_regex:
        log(f"      strip regex: {strip_regex}")
        log(f"      sources root: {sources_root}")
    else:
        log("      no stable common SF prefix found; keep original SF paths")

    run_cmd(
        [
            "info-process",
            "extract",
            "--coverage-type",
            "line",
            "--output",
            str(line_info),
            str(raw_all_info),
        ]
    )
    line_transform_args = [
        "info-process",
        "transform",
        "--normalize-paths",
        "--normalize-hit-counts",
    ]
    if strip_regex:
        line_transform_args.extend(["--strip-file-prefix", strip_regex])
    log(f"[5/9] Normalize line coverage -> {line_info}")
    run_cmd(line_transform_args + [str(line_info)])
    if sf_aliases:
        alias_hits = apply_sf_aliases(line_info, sf_aliases)
        if alias_hits:
            run_cmd(
                [
                    "info-process",
                    "transform",
                    "--normalize-paths",
                    "--normalize-hit-counts",
                    str(line_info),
                ]
            )
            dropped, mismatched = squash_duplicate_sf_headers(line_info)
            log(
                f"      applied SF aliases on line info (rewritten SF: {alias_hits}, "
                f"dropped extra SF: {dropped}, mismatched: {mismatched})"
            )
        else:
            log("      SF aliases configured for line info but no path matched")
    ensure_lf_lh_summary(line_info)

    shutil.copyfile(raw_toggle_info, toggle_info)
    toggle_name_map, mapping_conflicts = load_toggle_name_map(input_dats)
    renamed, unresolved = rewrite_toggle_brda_names(toggle_info, toggle_name_map)
    log(
        f"[6/9] Canonicalize toggle BRDA names in {toggle_info} "
        f"(rewritten: {renamed}, unresolved: {unresolved}, conflicts: {mapping_conflicts})"
    )
    toggle_transform_args = [
        "info-process",
        "transform",
        "--normalize-paths",
        "--normalize-hit-counts",
        "--set-block-ids",
    ]
    if strip_regex:
        toggle_transform_args.extend(["--strip-file-prefix", strip_regex])
    log(f"[7/9] Normalize toggle coverage -> {toggle_info}")
    run_cmd(
        toggle_transform_args + [str(toggle_info)]
    )
    if sf_aliases:
        alias_hits = apply_sf_aliases(toggle_info, sf_aliases)
        if alias_hits:
            run_cmd(
                [
                    "info-process",
                    "transform",
                    "--normalize-paths",
                    "--normalize-hit-counts",
                    "--set-block-ids",
                    str(toggle_info),
                ]
            )
            dropped, mismatched = squash_duplicate_sf_headers(toggle_info)
            log(
                f"      applied SF aliases on toggle info (rewritten SF: {alias_hits}, "
                f"dropped extra SF: {dropped}, mismatched: {mismatched})"
            )
        else:
            log("      SF aliases configured for toggle info but no path matched")

    coverage_files = [line_info, toggle_info]
    if has_user_coverage:
        shutil.copyfile(raw_user_info, user_info)
        user_transform_args = [
            "info-process",
            "transform",
            "--normalize-paths",
            "--normalize-hit-counts",
        ]
        if strip_regex:
            user_transform_args.extend(["--strip-file-prefix", strip_regex])
        log(f"[8/9] Normalize user coverage -> {user_info}")
        run_cmd(user_transform_args + [str(user_info)])
        if sf_aliases:
            alias_hits = apply_sf_aliases(user_info, sf_aliases)
            if alias_hits:
                run_cmd(
                    [
                        "info-process",
                        "transform",
                        "--normalize-paths",
                        "--normalize-hit-counts",
                        str(user_info),
                    ]
                )
                dropped, mismatched = squash_duplicate_sf_headers(user_info)
                log(
                    f"      applied SF aliases on user info (rewritten SF: {alias_hits}, "
                    f"dropped extra SF: {dropped}, mismatched: {mismatched})"
                )
            else:
                log("      SF aliases configured for user info but no path matched")
        coverage_files.append(user_info)
    else:
        log("[8/9] Skip user coverage normalization (no user records)")

    config_json.write_text("{}\n", encoding="utf-8")

    pack_args = [
        "info-process",
        "pack",
        "--output",
        str(output_zip),
        "--config",
        str(config_json),
        "--coverage-files",
        *(str(path) for path in coverage_files),
        "--generate-tables",
        "line",
    ]
    if sources_root:
        pack_args.extend(["--sources-root", sources_root])

    log(f"[9/9] Pack Coverview archive -> {output_zip}")
    run_cmd(pack_args)

    log(f"[OK] Done. Import {output_zip} into Coverview.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
