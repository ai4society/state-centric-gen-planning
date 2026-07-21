# Changelog

## v1.1.0 — Result artifacts consistent with ICAPS 2026 Table 1

Documents and preserves the correction to the committed result JSONs. Checkpoints
and the paper's reported numbers are unchanged.

### Fixed

- The result JSONs now use the size-dependent inference horizon
  `max(100, 10 * |O|)` from the paper (added in commit `e8a4fdc`, "step
  increase"). The result files released in `v1.0.0` used a fixed 100-step horizon,
  which truncated long plans on larger instances and depressed coverage in the
  extrapolation and large-instance cells; running
  `code/analysis/aggregate_results.py` over them reported below the published
  Table 1. The paper's printed numbers were unaffected — only the `v1.0.0`
  artifacts were stale.
- Aggregating the current `results/` over seeds 13/14/15 reproduces the published
  Table 1 in all 96 cells (mean and standard deviation).

### Preserved

- The pre-fix (`v1.0.0`) versions of the 158 corrected result files are archived
  under `changelog/v0-original-2026-03-14/`, with `NOTES.md` documenting the cause
  and the fix. Aggregating that folder reproduces the old (stale) coverage table.
- Release `v1.0.0` and its checkpoint assets are unchanged; the complete pre-fix
  state remains available at the `v1.0.0` tag.
