from __future__ import annotations

import fnmatch
import os
import re
from functools import lru_cache

from .command_utils import fail


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


def extract_tail_by_prefix_pattern(path: str, pattern: str) -> str | None:
    """Return path tail ('' or '/...') if path matches pattern as a prefix."""
    if not pattern:
        return None

    if "*" in pattern:
        matched = compile_wildcard_prefix_regex(pattern).match(path)
        if matched is None:
            return None
        return matched.group("tail") or ""

    if path == pattern:
        return ""
    prefix = "/" if pattern == "/" else pattern + "/"
    if path.startswith(prefix):
        return path[len(pattern) :]
    return None


def find_existing_rewrite_target(
    path: str,
    src: str,
    dst: str,
    allowed_targets: set[str],
) -> str | None:
    """
    Resolve alias by matching suffix under destination roots that already exist.

    This is needed when source and destination live under different launcher prefixes.
    """
    tail = extract_tail_by_prefix_pattern(path, src)
    if tail is None:
        return None

    candidates: list[str] = []
    for target in allowed_targets:
        target_tail = extract_tail_by_prefix_pattern(target, dst)
        if target_tail == tail:
            candidates.append(target)

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Prefer the candidate that shares the most path prefix with source.
    # If still ambiguous, fail to avoid silently merging to wrong target.
    scored = sorted(
        (
            (len(os.path.commonprefix([candidate, path])), candidate)
            for candidate in candidates
        ),
        reverse=True,
    )
    best_score = scored[0][0]
    best = sorted(candidate for score, candidate in scored if score == best_score)
    if len(best) == 1:
        return best[0]

    fail(
        "Ambiguous sf_alias target resolution for path "
        f"'{path}' via rule '{src} -> {dst}'. Candidates: {', '.join(best)}. "
        "Please make sf_alias patterns more specific."
    )


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


def validate_alias_pair(src: str, dst: str, context: str) -> None:
    src_wildcards = src.count("*")
    dst_wildcards = dst.count("*")
    if src_wildcards != dst_wildcards:
        fail(
            f"Invalid sf_alias rule '{context}': wildcard count must match "
            f"('{src}' -> '{dst}')"
        )


def validate_group_cycles(group_refs: dict[str, set[str]]) -> None:
    state: dict[str, int] = {}
    stack: list[str] = []
    index_by_group: dict[str, int] = {}

    def dfs(group: str) -> None:
        state[group] = 1
        index_by_group[group] = len(stack)
        stack.append(group)
        for ref in sorted(group_refs[group]):
            ref_state = state.get(ref, 0)
            if ref_state == 0:
                dfs(ref)
            elif ref_state == 1:
                cycle = stack[index_by_group[ref] :] + [ref]
                fail("Detected sf_alias group cycle: " + " -> ".join(cycle))
        stack.pop()
        index_by_group.pop(group, None)
        state[group] = 2

    for group in sorted(group_refs):
        if state.get(group, 0) == 0:
            dfs(group)


def parse_sf_alias_groups(raw_groups: dict[str, list[str]]) -> list[tuple[str, str]]:
    """
    Parse grouped sf_alias config.

    YAML format:
      sf_alias:
        group_name:
          - canonical_path
          - other_path_or_group_name
          - ...

    Each item is merged into the first item (canonical path).
    """
    if not raw_groups:
        return []

    group_names = set(raw_groups.keys())
    canonical_by_group: dict[str, str] = {}
    group_refs: dict[str, set[str]] = {name: set() for name in group_names}
    aliases: list[tuple[str, str]] = []

    for group, items in raw_groups.items():
        if not items:
            fail(f"Invalid sf_alias group '{group}': must contain at least one item")

        canonical_raw = items[0]
        canonical = normalize_coverage_path(canonical_raw)
        if not canonical:
            fail(f"Invalid sf_alias group '{group}': canonical path cannot be empty")
        if canonical in group_names:
            fail(
                f"Invalid sf_alias group '{group}': first item '{canonical_raw}' "
                "cannot be another group name"
            )
        canonical_by_group[group] = canonical

    for group, items in raw_groups.items():
        dst = canonical_by_group[group]
        for item in items[1:]:
            normalized_item = normalize_coverage_path(item)
            if not normalized_item:
                fail(f"Invalid sf_alias group '{group}': item cannot be empty")

            if normalized_item in group_names:
                ref_group = normalized_item
                group_refs[group].add(ref_group)
                src = canonical_by_group[ref_group]
                context = f"{group}<-{ref_group}"
            else:
                src = normalized_item
                context = f"{group}<-{item}"

            validate_alias_pair(src, dst, context)
            aliases.append((src, dst))

    validate_group_cycles(group_refs)

    source_to_target: dict[str, str] = {}
    deduped: list[tuple[str, str]] = []
    for src, dst in aliases:
        if src == dst:
            continue
        previous = source_to_target.get(src)
        if previous is not None and previous != dst:
            fail(
                "Conflicting sf_alias rules for source "
                f"'{src}': '{previous}' vs '{dst}'"
            )
        if previous is None:
            source_to_target[src] = dst
            deduped.append((src, dst))

    # Prefer more specific source patterns first (longer non-wildcard text wins).
    deduped.sort(key=lambda pair: len(pair[0].replace("*", "")), reverse=True)
    return deduped


def rewrite_sf_path_once_with_targets(
    path: str,
    aliases: list[tuple[str, str]],
    allowed_targets: set[str] | None,
) -> str:
    for src, dst in aliases:
        rewritten = rewrite_path_by_prefix(path, src, dst)
        if rewritten is not None and rewritten != path:
            if allowed_targets is None:
                return rewritten
            if rewritten in allowed_targets:
                return rewritten
            existing = find_existing_rewrite_target(path, src, dst, allowed_targets)
            if existing is not None and existing != path:
                return existing
            continue

        if allowed_targets is not None:
            existing = find_existing_rewrite_target(path, src, dst, allowed_targets)
            if existing is not None and existing != path:
                return existing
    return path


def rewrite_sf_path_chain(
    path: str,
    aliases: list[tuple[str, str]],
    allowed_targets: set[str] | None = None,
) -> list[str]:
    """Return transitive rewrite chain, starting with original path."""
    if not aliases:
        return [path]

    current = path
    chain = [current]
    index_by_path = {current: 0}

    while True:
        rewritten = rewrite_sf_path_once_with_targets(current, aliases, allowed_targets)
        if rewritten == current:
            break
        if rewritten in index_by_path:
            cycle = chain[index_by_path[rewritten] :] + [rewritten]
            fail("Detected sf_alias rewrite cycle: " + " -> ".join(cycle))
        current = rewritten
        index_by_path[current] = len(chain)
        chain.append(current)

    return chain


def rewrite_sf_path(
    path: str,
    aliases: list[tuple[str, str]],
    allowed_targets: set[str] | None = None,
) -> str:
    return rewrite_sf_path_chain(path, aliases, allowed_targets)[-1]


def parse_excluded_sf_paths(raw_paths: list[str]) -> list[str]:
    parsed_paths: list[str] = []
    for item in raw_paths:
        parsed = normalize_coverage_path(item)
        if not parsed:
            fail(f"Invalid --exclude-sf '{item}', path cannot be empty")
        parsed_paths.append(parsed)

    # Keep input order while removing duplicates.
    return list(dict.fromkeys(parsed_paths))
