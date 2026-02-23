from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

TOGGLE_INDEXED_PATTERN = re.compile(r"\[(\d+)\]:(0->1|1->0)$")
TOGGLE_SCALAR_PATTERN = re.compile(r":(0->1|1->0)$")
TOGGLE_COMMENT_PATTERN = re.compile(r"C '(.*)'\s+(-?\d+)$")


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
    mapping: dict[tuple[str, int], list[str]] = defaultdict(list)

    for raw in dat_path.read_bytes().splitlines():
        if not raw.startswith(b"C '"):
            continue

        text = raw.decode("latin1")
        if "\x01t\x02toggle" not in text:
            continue

        matched = TOGGLE_COMMENT_PATTERN.match(text)
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
