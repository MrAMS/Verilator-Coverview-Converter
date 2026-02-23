# Verilator Coverage Converter for Coverview

![](screenshots/p1.webp)

![](screenshots/p2.webp)

Convert one or more [Verilator](https://www.veripool.org/verilator/) `coverage.dat` files into a [Coverview](https://github.com/antmicro/coverview)-ready `.zip` archive.

## Features

- Merge multiple test runs into one dataset.
- Generate line / toggle / user coverage LCOV files.
- Perform hierarchical aggregation via grouped `sf_alias` with transitive propagation.
- Expand `exclude_sf` through effective alias relations (bi-directional + transitive).
- Use EDA-style point-level merge (hit-count accumulation + covered-if-hit).
- Validate alias model compatibility (DA/BRDA/FN point-set equality) and fail fast on mismatch.
- Dynamically remove redundant single-child folder prefixes for easier Coverview browsing.

## Requirements

Tools in `PATH`:
- `verilator_coverage` (from [verilator](https://verilator.org/), tested on `Verilator 5.040 2025-08-30 rev v5.040`)
- `info-process` (tested on [info-process@4c661cd](https://github.com/antmicro/info-process/commit/4c661cd6cb18df8ecfab7118cc0acbf4218b302a))

Python:
- `>=3.10`
- Use `uv` for environment/dependency management.

Coverview:
- Tested on [coverview@15386d3](https://github.com/antmicro/coverview/commit/15386d3b85712ab69e3a34cbc8a1ddd579cb14ad)

Setup:

```bash
uv sync
```

## Quick Start

```bash
uv run convert-coverage-to-coverview --args-yaml args.yaml
```

## CLI

Only YAML-driven configuration is supported.

```text
usage: convert-coverage-to-coverview [-h] --args-yaml FILE
```

Options:
- `--args-yaml FILE` (required): YAML config file.

## YAML Schema

Top-level keys:
- `input_dats` (string array, optional; default behavior is equivalent to `["coverage.dat"]`)
- `dataset` (string, optional; default `verilator`)
- `dats_root` (string, optional)
- `sf_alias` (mapping, optional)
- `exclude_sf` (string array, optional)

Example:

```yaml
dats_root: /path/to/bazel-testlogs/project
input_dats:
  - ru/TestPatternA/test.outputs/chiselsim/TestPatternA/PatternA.AND/random-stream/workdir-verilator/coverage.dat
  - ru/TestPatternA/test.outputs/chiselsim/TestPatternA/PatternA.OR/random-stream/workdir-verilator/coverage.dat
  - ru/TestPatternA/test.outputs/chiselsim/TestPatternA/PatternA.XOR/random-stream/workdir-verilator/coverage.dat
  - ru/TestPatternB/test.outputs/chiselsim/TestPatternB/PatternB.Max/random-stream/workdir-verilator/coverage.dat
  - ru/TestPatternB/test.outputs/chiselsim/TestPatternB/PatternB.Min/random-stream/workdir-verilator/coverage.dat
  - ru/TestPatternC/test.outputs/chiselsim/TestPatternC/PatternC/random-stream/workdir-verilator/coverage.dat
  - TestTop/test.outputs/chiselsim/TestTop/Top/upwards/workdir-verilator/coverage.dat
dataset: suite_demo
sf_alias:
  pattern_a:
    - "*/chiselsim/TestPatternA/PatternA.AND/random-stream"
    - "*/chiselsim/TestPatternA/PatternA.OR/random-stream"
    - "*/chiselsim/TestPatternA/PatternA.XOR/random-stream"
  pattern_b:
    - "*/chiselsim/TestPatternB/PatternB.Max/random-stream"
    - "*/chiselsim/TestPatternB/PatternB.Min/random-stream"
  top:
    - "*/chiselsim/TestTop/Top/upwards"
    - pattern_a
    - pattern_b
    - "*/chiselsim/TestPatternC/PatternC/random-stream"
exclude_sf:
  - "*/chiselsim/TestPatternA/PatternA.AND/random-stream/generated-sources/testbench.sv"
  - "*/chiselsim/TestPatternB/PatternB.Max/random-stream/generated-sources/testbench.sv"
  - "*/chiselsim/TestPatternC/PatternC/random-stream/generated-sources/testbench.sv"
```

## `sf_alias` Semantics

- `sf_alias` is a mapping: `group_name -> list`.
- First item of a list is the canonical target prefix.
- Remaining items can be:
  - path patterns, or
  - other group names.
- Group references are transitive.
- Wildcard `*` is supported; wildcard count must match per alias edge.
- Cycles are rejected.

## Merge Accuracy Rules

- **EDA-style point merge**: aliased records are merged by coverage point; hit counts are accumulated, and a point is covered if merged hit > 0.
- **Hierarchical alias graph**: `sf_alias` defines child-to-parent mapping with transitive propagation.
- **File-wise mapping**: mapping is resolved per concrete `SF` path.
- **Cross-layout safety**: for different launcher path layouts, alias target is resolved from existing `SF` entries in the current LCOV file.
- **Fail-fast mismatch guard**: before merge, source/target point models must match (`DA/BRDA/FN` set equality), otherwise conversion exits with explicit diffs.
- **Ambiguity protection**: if multiple equally plausible alias targets exist, conversion fails explicitly (no silent wrong merge).
- **Exclude consistency**: `exclude_sf` is expanded via effective alias edges in both directions and transitively.

## Dynamic Browse Path Compaction

After alias merge, the converter computes merged common prefix dynamically and strips only the redundant leading single-child chain.

This means Coverview opens closer to the first meaningful branch (for example `primary-sources/...`) without hardcoded directory-name rules.

## Outputs

For dataset `<name>`:

- `coverage_all_raw_<name>.info`
- `coverage_toggle_raw_<name>.info`
- `coverage_user_raw_<name>.info`
- `coverage_line_<name>.info`
- `coverage_toggle_<name>.info`
- `coverage_user_<name>.info` (only when user coverage exists)
- `coverview_config_<name>.json`
- `coverview_data_<name>.zip` (import this into Coverview)

## Pipeline Summary

1. Export raw LCOV (`combined`, `toggle`, `user`) from input `coverage.dat` files.
2. Compute initial path normalization prefix (`strip-file-prefix` + `sources-root`).
3. Process line coverage (extract, normalize, alias/exclude, recompute LF/LH).
4. Process toggle coverage (BRDA rename from raw comments, normalize, alias/exclude).
5. Process optional user coverage.
6. Dynamically compact merged redundant prefixes for browsing.
7. Pack Coverview archive.

## Project Layout

- `src/verilator_coverage_with_coverview/args_config.py`: YAML/CLI parsing.
- `src/verilator_coverage_with_coverview/path_alias.py`: alias graph, path rewrite, exclude expansion.
- `src/verilator_coverage_with_coverview/lcov.py`: LCOV transforms and dynamic merged-prefix compaction.
- `src/verilator_coverage_with_coverview/toggle_labels.py`: toggle label extraction and BRDA rewrite.
- `src/verilator_coverage_with_coverview/pipeline.py`: end-to-end orchestration.
- `src/verilator_coverage_with_coverview/cli.py`: console entrypoint.

## Notes

- If user coverage is absent, `coverage_user_<dataset>.info` is not packed.
- `info-process pack` warnings about missing `tests_*.desc` are expected unless test description files are provided.
