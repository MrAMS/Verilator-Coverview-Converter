from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .command_utils import TRANSFORM_BASE_ARGS, fail, log, run_cmd
from .path_alias import (
    matches_excluded_pattern,
    normalize_coverage_path,
    rewrite_sf_path_chain,
)


@dataclass(frozen=True)
class AliasApplyResult:
    rewritten_sf: int
    validated_alias_pairs: int
    alias_map: dict[str, str]


@dataclass(frozen=True)
class SfPointSignature:
    da_lines: frozenset[int]
    brda_points: frozenset[tuple[int, str]]
    fn_points: frozenset[tuple[int, str]]


@dataclass
class SfCoverageBucket:
    tn_payload: str | None = None
    da_hits: dict[int, int] = field(default_factory=dict)
    brda_hits: dict[tuple[int, str, str], int] = field(default_factory=dict)
    fn_lines: dict[str, int] = field(default_factory=dict)
    fnda_hits: dict[str, int] = field(default_factory=dict)
    keep_lf_lh: bool = False
    keep_brf_brh: bool = False
    keep_fnf_fnh: bool = False


def resolve_sf_alias_map(
    known_paths: set[str],
    aliases: list[tuple[str, str]],
) -> dict[str, str]:
    mapping: dict[str, str] = {}

    for path in sorted(known_paths):
        chain = rewrite_sf_path_chain(path, aliases, allowed_targets=known_paths)
        mapping[path] = chain[-1]

    return mapping


def collect_sf_signatures(lines: list[str]) -> dict[str, SfPointSignature]:
    signatures: dict[str, dict[str, set]] = {}
    current_sf = ""
    current_da: set[int] = set()
    current_brda: set[tuple[int, str]] = set()
    current_fn: set[tuple[int, str]] = set()

    def flush_current() -> None:
        nonlocal current_sf, current_da, current_brda, current_fn
        if not current_sf:
            return
        bucket = signatures.setdefault(
            current_sf,
            {"da": set(), "brda": set(), "fn": set()},
        )
        bucket["da"].update(current_da)
        bucket["brda"].update(current_brda)
        bucket["fn"].update(current_fn)
        current_sf = ""
        current_da = set()
        current_brda = set()
        current_fn = set()

    for line in lines:
        if line.startswith("SF:"):
            flush_current()
            current_sf = normalize_coverage_path(line[3:])
            continue

        if not current_sf:
            continue

        if line.startswith("DA:"):
            parts = line[3:].strip().split(",", 1)
            if len(parts) == 2:
                try:
                    current_da.add(int(parts[0]))
                except ValueError:
                    pass
            continue

        if line.startswith("BRDA:"):
            parts = line[5:].strip().split(",", 3)
            if len(parts) == 4:
                try:
                    line_no = int(parts[0])
                except ValueError:
                    line_no = None
                if line_no is not None:
                    current_brda.add((line_no, parts[2]))
            continue

        if line.startswith("FN:"):
            parts = line[3:].strip().split(",", 1)
            if len(parts) == 2:
                try:
                    line_no = int(parts[0])
                except ValueError:
                    line_no = None
                if line_no is not None:
                    current_fn.add((line_no, parts[1]))
            continue

        if line.startswith("end_of_record"):
            flush_current()

    flush_current()
    return {
        sf: SfPointSignature(
            da_lines=frozenset(payload["da"]),
            brda_points=frozenset(payload["brda"]),
            fn_points=frozenset(payload["fn"]),
        )
        for sf, payload in signatures.items()
    }


def format_signature_diff(
    src_path: str,
    dst_path: str,
    src_signature: SfPointSignature,
    dst_signature: SfPointSignature,
) -> str:
    def summarize(label: str, src_set: set, dst_set: set) -> str:
        src_only = sorted(src_set - dst_set)
        dst_only = sorted(dst_set - src_set)
        preview_limit = 3
        src_preview = ", ".join(str(item) for item in src_only[:preview_limit]) or "-"
        dst_preview = ", ".join(str(item) for item in dst_only[:preview_limit]) or "-"
        return (
            f"{label} src={len(src_set)} dst={len(dst_set)} "
            f"src_only={len(src_only)}[{src_preview}] "
            f"dst_only={len(dst_only)}[{dst_preview}]"
        )

    return "\n".join(
        [
            f"  source: {src_path}",
            f"  target: {dst_path}",
            "  " + summarize(
                "DA",
                set(src_signature.da_lines),
                set(dst_signature.da_lines),
            ),
            "  " + summarize(
                "BRDA",
                set(src_signature.brda_points),
                set(dst_signature.brda_points),
            ),
            "  " + summarize(
                "FN",
                set(src_signature.fn_points),
                set(dst_signature.fn_points),
            ),
        ]
    )


def validate_alias_model_compatibility(
    alias_map: dict[str, str],
    signatures: dict[str, SfPointSignature],
    coverage_label: str,
    skip_paths: set[str] | None = None,
) -> int:
    skipped = skip_paths or set()
    mismatches: list[str] = []
    validated = 0
    for src, dst in sorted(alias_map.items()):
        if src == dst:
            continue
        if src in skipped or dst in skipped:
            continue
        src_signature = signatures.get(src)
        dst_signature = signatures.get(dst)
        if src_signature is None or dst_signature is None:
            mismatches.append(
                "\n".join(
                    [
                        f"  source: {src}",
                        f"  target: {dst}",
                        "  missing signature in LCOV record",
                    ]
                )
            )
            continue
        if src_signature != dst_signature:
            mismatches.append(format_signature_diff(src, dst, src_signature, dst_signature))
            continue
        validated += 1

    if mismatches:
        details = "\n\n".join(mismatches[:8])
        extra = "" if len(mismatches) <= 8 else f"\n\n... and {len(mismatches) - 8} more mismatches"
        fail(
            "sf_alias point-level merge validation failed: source and target coverage models differ.\n"
            f"coverage type: {coverage_label}\n"
            "EDA-style merge requires identical DA/BRDA/FN point sets between aliased records.\n"
            "Please regenerate consistent simulation outputs or adjust your sf_alias groups.\n\n"
            + details
            + extra
        )
    return validated


def apply_sf_aliases(
    info_file: Path,
    aliases: list[tuple[str, str]],
    coverage_label: str,
    skip_validation_paths: set[str] | None = None,
    precomputed_alias_map: dict[str, str] | None = None,
) -> AliasApplyResult:
    if not aliases:
        return AliasApplyResult(rewritten_sf=0, validated_alias_pairs=0, alias_map={})

    lines = info_file.read_text(encoding="utf-8").splitlines(keepends=True)
    if precomputed_alias_map is None:
        known_paths = {normalize_coverage_path(line[3:]) for line in lines if line.startswith("SF:")}
        alias_map = resolve_sf_alias_map(known_paths, aliases)
    else:
        alias_map = precomputed_alias_map
    signatures = collect_sf_signatures(lines)
    validated_pairs = validate_alias_model_compatibility(
        alias_map,
        signatures,
        coverage_label,
        skip_paths=skip_validation_paths,
    )

    rewritten_sf = 0
    output: list[str] = []
    for line in lines:
        if line.startswith("SF:"):
            old_path = normalize_coverage_path(line[3:])
            new_path = alias_map.get(old_path, old_path)
            if new_path != old_path:
                rewritten_sf += 1
            output.append(f"SF:{new_path}\n")
            continue
        output.append(line)

    info_file.write_text("".join(output), encoding="utf-8")
    return AliasApplyResult(
        rewritten_sf=rewritten_sf,
        validated_alias_pairs=validated_pairs,
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


def parse_lcov_hits(raw_value: str) -> int:
    value = raw_value.strip()
    if value == "-":
        return 0
    try:
        return int(value)
    except ValueError:
        return 0


def merge_lcov_records_by_sf(info_file: Path) -> tuple[int, int]:
    """
    Merge duplicate SF records with EDA-style point accumulation.

    - DA/FNDA/BRDA hit counts are summed.
    - Covered status naturally follows merged hit count > 0.
    """
    buckets: dict[str, SfCoverageBucket] = {}
    order: list[str] = []

    input_records = 0
    input_points = 0

    current_tn: str | None = None
    current_sf = ""
    current_da: dict[int, int] = {}
    current_brda: dict[tuple[int, str, str], int] = {}
    current_fn_lines: dict[str, int] = {}
    current_fnda: dict[str, int] = {}
    current_keep_lf_lh = False
    current_keep_brf_brh = False
    current_keep_fnf_fnh = False

    def reset_record_state() -> None:
        nonlocal current_tn, current_sf
        nonlocal current_da, current_brda, current_fn_lines, current_fnda
        nonlocal current_keep_lf_lh, current_keep_brf_brh, current_keep_fnf_fnh
        current_tn = None
        current_sf = ""
        current_da = {}
        current_brda = {}
        current_fn_lines = {}
        current_fnda = {}
        current_keep_lf_lh = False
        current_keep_brf_brh = False
        current_keep_fnf_fnh = False

    def flush_record() -> None:
        nonlocal input_records, input_points
        if not current_sf:
            reset_record_state()
            return

        input_records += 1
        input_points += len(current_da) + len(current_brda) + len(current_fnda)

        bucket = buckets.get(current_sf)
        if bucket is None:
            bucket = SfCoverageBucket()
            buckets[current_sf] = bucket
            order.append(current_sf)

        if bucket.tn_payload is None and current_tn is not None:
            bucket.tn_payload = current_tn

        for line_no, hits in current_da.items():
            bucket.da_hits[line_no] = bucket.da_hits.get(line_no, 0) + hits
        for point, hits in current_brda.items():
            bucket.brda_hits[point] = bucket.brda_hits.get(point, 0) + hits
        for name, line_no in current_fn_lines.items():
            bucket.fn_lines.setdefault(name, line_no)
        for name, hits in current_fnda.items():
            bucket.fnda_hits[name] = bucket.fnda_hits.get(name, 0) + hits

        bucket.keep_lf_lh = bucket.keep_lf_lh or current_keep_lf_lh
        bucket.keep_brf_brh = bucket.keep_brf_brh or current_keep_brf_brh
        bucket.keep_fnf_fnh = bucket.keep_fnf_fnh or current_keep_fnf_fnh
        reset_record_state()

    reset_record_state()

    with info_file.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if raw_line.startswith("TN:"):
                current_tn = raw_line[3:].rstrip("\n")
                continue
            if raw_line.startswith("SF:"):
                current_sf = normalize_coverage_path(raw_line[3:])
                continue
            if raw_line.startswith("DA:"):
                parts = line[3:].split(",", 1)
                if len(parts) == 2:
                    try:
                        line_no = int(parts[0])
                    except ValueError:
                        line_no = None
                    if line_no is not None:
                        current_da[line_no] = current_da.get(line_no, 0) + parse_lcov_hits(parts[1])
                continue
            if raw_line.startswith("BRDA:"):
                parts = line[5:].split(",", 3)
                if len(parts) == 4:
                    try:
                        line_no = int(parts[0])
                    except ValueError:
                        line_no = None
                    if line_no is not None:
                        key = (line_no, parts[1], parts[2])
                        current_brda[key] = current_brda.get(key, 0) + parse_lcov_hits(parts[3])
                continue
            if raw_line.startswith("FN:"):
                parts = line[3:].split(",", 1)
                if len(parts) == 2:
                    try:
                        line_no = int(parts[0])
                    except ValueError:
                        line_no = None
                    if line_no is not None:
                        current_fn_lines.setdefault(parts[1], line_no)
                continue
            if raw_line.startswith("FNDA:"):
                parts = line[5:].split(",", 1)
                if len(parts) == 2:
                    current_fnda[parts[1]] = current_fnda.get(parts[1], 0) + parse_lcov_hits(parts[0])
                continue
            if raw_line.startswith("LF:") or raw_line.startswith("LH:"):
                current_keep_lf_lh = True
                continue
            if raw_line.startswith("BRF:") or raw_line.startswith("BRH:"):
                current_keep_brf_brh = True
                continue
            if raw_line.startswith("FNF:") or raw_line.startswith("FNH:"):
                current_keep_fnf_fnh = True
                continue
            if raw_line.startswith("end_of_record"):
                flush_record()
                continue

    flush_record()

    output: list[str] = []
    output_points = 0
    for sf in order:
        bucket = buckets[sf]
        if bucket.tn_payload is not None:
            output.append(f"TN:{bucket.tn_payload}\n")
        output.append(f"SF:{sf}\n")

        for name, line_no in sorted(bucket.fn_lines.items(), key=lambda item: (item[1], item[0])):
            output.append(f"FN:{line_no},{name}\n")
        for name in sorted(bucket.fnda_hits):
            output.append(f"FNDA:{bucket.fnda_hits[name]},{name}\n")
            output_points += 1

        function_names = set(bucket.fn_lines) | set(bucket.fnda_hits)
        if function_names or bucket.keep_fnf_fnh:
            fnf = len(function_names)
            fnh = sum(1 for name in function_names if bucket.fnda_hits.get(name, 0) > 0)
            output.append(f"FNF:{fnf}\n")
            output.append(f"FNH:{fnh}\n")

        for line_no in sorted(bucket.da_hits):
            output.append(f"DA:{line_no},{bucket.da_hits[line_no]}\n")
            output_points += 1
        if bucket.da_hits or bucket.keep_lf_lh:
            lf = len(bucket.da_hits)
            lh = sum(1 for hits in bucket.da_hits.values() if hits > 0)
            output.append(f"LF:{lf}\n")
            output.append(f"LH:{lh}\n")

        for line_no, block_id, name in sorted(bucket.brda_hits):
            output.append(f"BRDA:{line_no},{block_id},{name},{bucket.brda_hits[(line_no, block_id, name)]}\n")
            output_points += 1
        if bucket.brda_hits or bucket.keep_brf_brh:
            brf = len(bucket.brda_hits)
            brh = sum(1 for hits in bucket.brda_hits.values() if hits > 0)
            output.append(f"BRF:{brf}\n")
            output.append(f"BRH:{brh}\n")

        output.append("end_of_record\n")

    info_file.write_text("".join(output), encoding="utf-8")
    merged_records = input_records - len(order)
    collapsed_points = input_points - output_points
    return merged_records, collapsed_points


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
) -> None:
    """Apply SF alias merge + SF exclusion to one normalized coverage file."""
    sf_paths = collect_sf_paths(info_file)
    identity_map = {path: path for path in sf_paths}
    alias_map = identity_map
    if sf_aliases:
        alias_map = resolve_sf_alias_map(set(sf_paths), sf_aliases)

    effective_excludes: set[str] = set()
    if excluded_sf_patterns:
        effective_excludes = expand_excluded_paths_for_alias_map(excluded_sf_patterns, alias_map)

    if sf_aliases:
        alias_result = apply_sf_aliases(
            info_file,
            sf_aliases,
            coverage_label,
            skip_validation_paths=effective_excludes,
            precomputed_alias_map=alias_map,
        )
        alias_map = alias_result.alias_map
        if alias_result.rewritten_sf:
            merged_records, collapsed_points = merge_lcov_records_by_sf(info_file)
            log(
                f"      applied SF aliases on {coverage_label} info (rewritten SF: {alias_result.rewritten_sf}, "
                f"validated pairs: {alias_result.validated_alias_pairs}, "
                f"merged records: {merged_records}, collapsed points: {collapsed_points})"
            )
        else:
            log(
                f"      SF aliases configured for {coverage_label} info but no path matched "
                f"(validated pairs: {alias_result.validated_alias_pairs})"
            )

    if effective_excludes:
        removed_records, removed_files = drop_excluded_sf_records(info_file, effective_excludes)
        log(
            f"      excluded {coverage_label} coverage records: {removed_records} "
            f"(matched SF files: {removed_files})"
        )
