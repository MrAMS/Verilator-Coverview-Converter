from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .command_utils import TRANSFORM_BASE_ARGS, log, run_cmd
from .path_alias import (
    matches_excluded_pattern,
    normalize_coverage_path,
    rewrite_sf_path_chain,
)


@dataclass(frozen=True)
class AliasApplyResult:
    rewritten_sf: int
    dropped_parent_records: int
    alias_map: dict[str, str]


def resolve_sf_alias_map(
    known_paths: set[str],
    aliases: list[tuple[str, str]],
) -> dict[str, str]:
    mapping: dict[str, str] = {}

    for path in sorted(known_paths):
        chain = rewrite_sf_path_chain(path, aliases, allowed_targets=known_paths)
        mapping[path] = chain[-1]

    return mapping


def collect_shadowed_targets(alias_map: dict[str, str]) -> set[str]:
    """
    Collect target SFs that should drop their parent record.

    In hierarchical aggregation mode, if any child path maps to target T, then
    original record on T is dropped and replaced by aggregated child coverage.
    """
    return {dst for src, dst in alias_map.items() if src != dst}


def apply_sf_aliases(info_file: Path, aliases: list[tuple[str, str]]) -> AliasApplyResult:
    if not aliases:
        return AliasApplyResult(rewritten_sf=0, dropped_parent_records=0, alias_map={})

    lines = info_file.read_text(encoding="utf-8").splitlines(keepends=True)
    known_paths = {normalize_coverage_path(line[3:]) for line in lines if line.startswith("SF:")}
    alias_map = resolve_sf_alias_map(known_paths, aliases)
    shadowed_targets = collect_shadowed_targets(alias_map)

    rewritten_sf = 0
    dropped_parent_records = 0
    output: list[str] = []
    record: list[str] = []
    record_sf = ""

    def flush_record() -> None:
        nonlocal rewritten_sf, dropped_parent_records, record, record_sf
        if not record:
            return

        normalized_sf = normalize_coverage_path(record_sf)
        if normalized_sf and normalized_sf in shadowed_targets:
            dropped_parent_records += 1
            record = []
            record_sf = ""
            return

        for item in record:
            if not item.startswith("SF:"):
                output.append(item)
                continue
            old_path = normalize_coverage_path(item[3:])
            new_path = alias_map.get(old_path, old_path)
            if new_path != old_path:
                rewritten_sf += 1
            output.append(f"SF:{new_path}\n")

        record = []
        record_sf = ""

    for line in lines:
        record.append(line)
        if line.startswith("SF:"):
            record_sf = line[3:].strip()
            continue
        if line.startswith("end_of_record"):
            flush_record()

    flush_record()

    info_file.write_text("".join(output), encoding="utf-8")
    return AliasApplyResult(
        rewritten_sf=rewritten_sf,
        dropped_parent_records=dropped_parent_records,
        alias_map=alias_map,
    )


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
                and any(matches_excluded_pattern(current_sf, pattern) for pattern in patterns)
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


def expand_excluded_paths_for_alias_map(
    base_patterns: list[str], alias_map: dict[str, str]
) -> set[str]:
    """
    Expand exclude patterns through effective alias edges for this LCOV file.

    Expansion is performed only on aliases discovered in this LCOV file, so
    unrelated configured alias edges do not over-exclude.
    """
    if not base_patterns:
        return set()

    nodes: set[str] = set(alias_map.keys()) | set(alias_map.values())
    if not nodes:
        return set()

    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for src, dst in alias_map.items():
        if src == dst:
            continue
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    matched = {
        node
        for node in nodes
        if any(matches_excluded_pattern(node, pattern) for pattern in base_patterns)
    }
    if not matched:
        return set()

    expanded = set(matched)
    queue = list(matched)
    while queue:
        current = queue.pop()
        for neighbor in adjacency[current]:
            if neighbor in expanded:
                continue
            expanded.add(neighbor)
            queue.append(neighbor)
    return expanded


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


def compute_post_merge_strip_prefix(info_files: list[Path]) -> str:
    """
    Compute extra strip prefix after alias merge to improve Coverview tree browsing.

    Dynamic rule: remove only the leading folders that are guaranteed redundant
    (single-child chain), and keep the first branching folder visible.
    """
    sf_paths: list[str] = []
    for info_file in info_files:
        sf_paths.extend(collect_sf_paths(info_file))
    if not sf_paths:
        return ""

    is_abs_list = [path.startswith("/") for path in sf_paths]
    if any(is_abs_list) and not all(is_abs_list):
        return ""

    common_prefix = normalize_coverage_path(os.path.commonpath(sf_paths))
    if common_prefix in ("", "/"):
        return ""

    common_prefix = common_prefix.rstrip("/")
    strip_prefix = os.path.dirname(common_prefix)
    if strip_prefix in ("", ".", "/"):
        return ""
    return normalize_coverage_path(strip_prefix)


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
    excluded_sf_patterns: list[str],
    set_block_ids: bool = False,
) -> None:
    """Apply SF alias merge + SF exclusion to one normalized coverage file."""
    alias_map: dict[str, str] = {}
    if sf_aliases:
        alias_result = apply_sf_aliases(info_file, sf_aliases)
        alias_map = alias_result.alias_map
        if alias_result.rewritten_sf or alias_result.dropped_parent_records:
            # Re-run transform after alias rewrite so info-process can merge records.
            transform_info_file(info_file, strip_regex="", set_block_ids=set_block_ids)
            dropped, mismatched = squash_duplicate_sf_headers(info_file)
            log(
                f"      applied SF aliases on {coverage_label} info (rewritten SF: {alias_result.rewritten_sf}, "
                f"dropped parent records: {alias_result.dropped_parent_records}, "
                f"dropped extra SF: {dropped}, mismatched: {mismatched})"
            )
        else:
            log(f"      SF aliases configured for {coverage_label} info but no path matched")

    if excluded_sf_patterns:
        if not alias_map:
            sf_paths = collect_sf_paths(info_file)
            alias_map = {path: path for path in sf_paths}
        effective_excludes = expand_excluded_paths_for_alias_map(excluded_sf_patterns, alias_map)
        removed_records, removed_files = drop_excluded_sf_records(info_file, effective_excludes)
        log(
            f"      excluded {coverage_label} coverage records: {removed_records} "
            f"(matched SF files: {removed_files})"
        )
