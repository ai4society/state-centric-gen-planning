# Original results (v1.0.0, 2026-03-14) — superseded

This folder preserves the **pre-fix versions of the 158 result files** that were
later corrected, exactly as released in the `v1.0.0` tag. It is kept for
provenance only. **Do not use these numbers**; the corrected results live in
`results/` on the main branch.

## What was wrong

These JSONs were generated with a fixed 100-step inference horizon. The paper
uses a size-dependent horizon of `max(100, 10 * |O|)` steps (Inference and
Metrics section), where `|O|` is the number of objects in the instance. The fixed
cap truncated long plans on larger instances, so failures accumulated at exactly
`plan_len == 100` with `val_executable == true` (the partial plan was legal under
VAL but had not yet reached the goal). Running `code/analysis/aggregate_results.py`
over this snapshot therefore reports coverage below the published Table 1 in the
extrapolation and large-instance cells.

The paper's printed numbers are correct. Only these `v1.0.0` artifacts were stale:
they predate the horizon fix.

## The fix

Commit `e8a4fdc` ("step increase", 2026-03-17) added the size-dependent horizon
to both inference scripts:

```python
num_objects = len(prob.objects) + len(dom.constants)
dynamic_max_steps = max(args.max_steps, 10 * num_objects)
```

Between the code state that produced this snapshot (`e8a4fdc~1`) and the fix, the
horizon line is the only behavioral change to `inference_lstm.py` and
`inference_xgb.py`. Verify with:

```bash
git diff e8a4fdc~1 e8a4fdc -- code/modeling/inference_lstm.py code/modeling/inference_xgb.py
```

The corrected results were then regenerated with the fixed horizon (committed to
`main` in "ICAPS official results"). Aggregating the current `results/` over seeds
13/14/15 reproduces the published Table 1 in all 96 cells, mean and standard
deviation.

## Reproducing the old (stale) numbers

This snapshot is the complete set of files that changed, so it is self-contained:

```bash
python -m code.analysis.aggregate_results --results_dir changelog/v0-original-2026-03-14/results
```

That prints the pre-fix coverage table. Equivalently, aggregate the `v1.0.0` tag.
See `../../CHANGELOG.md` for the summary.
