from __future__ import annotations

import os
import re
import shutil

from .args_config import parse_args, resolve_inputs_and_dataset
from .command_utils import log, need_cmd, run_cmd
from .lcov import (
    apply_alias_and_exclusions,
    compute_post_merge_strip_prefix,
    compute_path_rewrite,
    ensure_lf_lh_summary,
    export_coverage_info,
    has_sf_records,
    transform_info_file,
)
from .models import build_output_files
from .path_alias import parse_excluded_sf_paths, parse_sf_alias_groups
from .toggle_labels import load_toggle_name_map, rewrite_toggle_brda_names


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Parse alias/exclude intent once, then reuse across line/toggle/user stages.
    sf_aliases = parse_sf_alias_groups(args.sf_alias)
    excluded_sf_patterns = parse_excluded_sf_paths(args.exclude_sf)
    if excluded_sf_patterns:
        log(f"[0/10] Exclude SF patterns: {len(excluded_sf_patterns)}")
        for path in excluded_sf_patterns:
            log(f"      - {path}")

    input_dats, dataset = resolve_inputs_and_dataset(args)
    outputs = build_output_files(dataset)

    need_cmd("verilator_coverage")
    need_cmd("info-process")

    # Stage 1: export raw LCOV data from Verilator.
    log(
        f"[1/10] Export combined coverage from {len(input_dats)} dat file(s) -> "
        f"{outputs.raw_all_info}"
    )
    export_coverage_info(input_dats, outputs.raw_all_info, filter_type=None)

    log(
        f"[2/10] Extract toggle coverage from {len(input_dats)} dat file(s) -> "
        f"{outputs.raw_toggle_info}"
    )
    export_coverage_info(input_dats, outputs.raw_toggle_info, filter_type="toggle")

    log(
        f"[3/10] Extract user coverage from {len(input_dats)} dat file(s) -> "
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

    log("[4/10] Analyze SF paths for prefix stripping")
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
    log(f"[5/10] Normalize line coverage -> {outputs.line_info}")
    transform_info_file(outputs.line_info, strip_regex=strip_regex)
    apply_alias_and_exclusions(
        outputs.line_info,
        coverage_label="line",
        sf_aliases=sf_aliases,
        excluded_sf_patterns=excluded_sf_patterns,
    )
    ensure_lf_lh_summary(outputs.line_info)

    # Stage 4: toggle coverage pipeline (with BRDA label rewrite).
    shutil.copyfile(outputs.raw_toggle_info, outputs.toggle_info)
    toggle_name_map, mapping_conflicts = load_toggle_name_map(input_dats)
    renamed, unresolved = rewrite_toggle_brda_names(outputs.toggle_info, toggle_name_map)
    log(
        f"[6/10] Canonicalize toggle BRDA names in {outputs.toggle_info} "
        f"(rewritten: {renamed}, unresolved: {unresolved}, conflicts: {mapping_conflicts})"
    )
    log(f"[7/10] Normalize toggle coverage -> {outputs.toggle_info}")
    transform_info_file(outputs.toggle_info, strip_regex=strip_regex, set_block_ids=True)
    apply_alias_and_exclusions(
        outputs.toggle_info,
        coverage_label="toggle",
        sf_aliases=sf_aliases,
        excluded_sf_patterns=excluded_sf_patterns,
        set_block_ids=True,
    )

    # Stage 5: optional user coverage pipeline.
    coverage_files = [outputs.line_info, outputs.toggle_info]
    if has_user_coverage:
        shutil.copyfile(outputs.raw_user_info, outputs.user_info)
        log(f"[8/10] Normalize user coverage -> {outputs.user_info}")
        transform_info_file(outputs.user_info, strip_regex=strip_regex)
        apply_alias_and_exclusions(
            outputs.user_info,
            coverage_label="user",
            sf_aliases=sf_aliases,
            excluded_sf_patterns=excluded_sf_patterns,
        )
        coverage_files.append(outputs.user_info)
    else:
        log("[8/10] Skip user coverage normalization (no user records)")

    # Stage 6: compact merged common prefix for easier Coverview browsing.
    post_merge_strip_prefix = compute_post_merge_strip_prefix(coverage_files)
    if post_merge_strip_prefix:
        post_merge_strip_regex = re.escape(post_merge_strip_prefix.rstrip("/")) + "/?"
        log("[9/10] Compact merged SF prefix for browsing")
        log(f"      extra strip prefix: {post_merge_strip_prefix}")
        transform_info_file(outputs.line_info, strip_regex=post_merge_strip_regex)
        transform_info_file(outputs.toggle_info, strip_regex=post_merge_strip_regex, set_block_ids=True)
        if has_user_coverage:
            transform_info_file(outputs.user_info, strip_regex=post_merge_strip_regex)

        if sources_root:
            if os.path.isabs(post_merge_strip_prefix):
                sources_root = post_merge_strip_prefix
            else:
                sources_root = os.path.normpath(os.path.join(sources_root, post_merge_strip_prefix))
            log(f"      adjusted sources root: {sources_root}")
        else:
            log("      sources root unavailable, skip sources-root adjustment")
    else:
        log("[9/10] No additional merged SF prefix to compact")

    # Stage 7: pack Coverview archive.
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

    log(f"[10/10] Pack Coverview archive -> {outputs.output_zip}")
    run_cmd(pack_args)

    log(f"[OK] Done. Import {outputs.output_zip} into Coverview.")
    return 0
