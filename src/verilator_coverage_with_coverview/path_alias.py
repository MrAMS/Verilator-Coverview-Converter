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
            fail(f"Invalid --sf-alias '{item}', expected FROM=TO")
        src, dst = item.split("=", 1)
        src = normalize_coverage_path(src)
        dst = normalize_coverage_path(dst)
        if not src:
            fail(f"Invalid --sf-alias '{item}', FROM cannot be empty")
        src_wildcards = src.count("*")
        dst_wildcards = dst.count("*")
        if src_wildcards != dst_wildcards:
            fail(f"Invalid --sf-alias '{item}', wildcard count must match on both sides")
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


def parse_excluded_sf_paths(raw_paths: list[str]) -> list[str]:
    parsed_paths: list[str] = []
    for item in raw_paths:
        parsed = normalize_coverage_path(item)
        if not parsed:
            fail(f"Invalid --exclude-sf '{item}', path cannot be empty")
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
