# Evaluation Methodology

## 1. Overview

This framework separates concerns into three layers so it stays modular
and reusable across future models:

1. **Metrics** (`eval_framework/cv/detection_metrics.py`,
   `eval_framework/llm/llm_metrics.py`) — pure functions, no I/O, no timing.
2. **Runners** (`eval_framework/cv/runner.py`, `eval_framework/llm/runner.py`)
   — wrap a user-supplied model function, add timing/FPS, and assemble a
   uniform `EvalResult`.
3. **Core** (`eval_framework/core/`) — timing utilities, result
   persistence/versioning, and config loading, shared by both CV and LLM.

Because the runners only depend on a plain Python function
(`predict_fn` / `generate_fn`) with a fixed input/output shape, swapping
in a different or newer model version requires no changes to the
metrics or storage code — only a new function pointed at the new model.

## 2. Computer Vision metrics

| Metric | Definition | Notes |
|---|---|---|
| Precision | TP / (TP + FP) | Computed globally across all images/classes at the given IoU threshold |
| Recall | TP / (TP + FN) | Same matching as above |
| F1 Score | 2·P·R / (P + R) | Harmonic mean of the above |
| mAP | Mean of per-class Average Precision | 11-point interpolated AP (PASCAL VOC style), averaged over all classes present in ground truth |
| FPS | 1000 / mean_latency_ms | Derived from the same timed inference calls used for latency |
| Inference latency | Wall-clock time per `predict_fn` call | Mean, p50 (median), p95, p99, min, max reported; first `warmup_runs` calls excluded from timing to avoid cold-start skew |

**Matching**: for each image and class, predictions are sorted by
confidence descending and greedily matched to the highest-IoU unmatched
ground-truth box; a match counts as a true positive if IoU ≥
`iou_threshold` (default 0.5, configurable).

**Reference implementation**: the detection metrics are implemented from
first principles (no external CV library required), so the framework
runs anywhere Python runs. For large-scale production benchmarking
against COCO-format datasets, the same public functions
(`precision_recall_f1`, `mean_average_precision`) can be replaced with
calls to `pycocotools` without touching the runner or CLI scripts.

## 3. LLM / AI system metrics

| Metric | Definition | Notes |
|---|---|---|
| Accuracy | Exact match (normalized) and/or token-overlap F1 against a reference answer | Marked "not applicable" if no `reference` is supplied in any test case |
| Relevance | Fraction of the query's meaningful tokens reflected in the response | Lexical-overlap proxy; see "Upgrading metrics" below |
| Groundedness | Fraction of the response's tokens also present in the supplied context | Marked "not applicable" if no `context` is supplied |
| Hallucination rate | Fraction of responses whose groundedness falls below `hallucination_threshold` (default 0.5) | Inverse-of-groundedness at the response level |
| Latency | Wall-clock time per `generate_fn` call | Mean, p95, p99 reported |
| Token usage / cost | Input/output token counts and estimated USD cost | Uses real token counts if `generate_fn` returns them; otherwise a word-count based estimate, and the result is flagged `tokens_estimated: true` |

**Handling "not applicable" metrics**: rather than silently defaulting
missing-input metrics to 0 or 1 (which would be misleading), the
framework records them in a separate `not_applicable` dict with a plain
reason, e.g. `"No context provided in test cases."`. This keeps results
honest and auditable.

### Upgrading metrics (LLM-as-judge)

The default Accuracy/Relevance/Groundedness implementations are
deterministic lexical-overlap proxies — chosen so the framework runs
with zero setup (no API key, no embedding model). For higher-fidelity
scoring, `eval_framework/llm/judge.py` provides an `LLMJudge` template
that scores the same three dimensions using an actual model call. Swap
it in by scoring with `LLMJudge().score(...)` instead of the lexical
functions inside a custom runner — the output shape (floats in [0, 1])
is unchanged, so `EvalResult` and comparison/storage code need no edits.

## 4. Reproducibility

Every run captures, inside the saved `EvalResult`:
- `config` — the exact settings used (IoU threshold, seed, dataset path, hallucination threshold, etc.)
- `environment` — Python version, platform, and git commit (if run inside a git repo)
- `timestamp` — UTC ISO-8601

To reproduce a specific past result:
1. Open its JSON file under `results/`.
2. Re-run the corresponding script with the same `config` values and the same dataset file.
3. For the bundled mock models, pass the same `--seed` to get identical synthetic behavior.

For a **real** model, reproducibility additionally depends on: pinning
the exact model version/checkpoint, using the same dataset file
(consider hashing it), and running on comparable hardware for
latency/FPS comparisons (latency numbers are not portable across
different machines).

## 5. Comparing model versions

`scripts/compare_models.py` reads `results/history.jsonl` (append-only
log of every run) and pulls the latest run per requested model name,
tabulating shared metrics side by side. This is how "compare different
model versions/configurations" (a required deliverable) is satisfied —
just give each version/config run a distinct `--model-name`, e.g.
`yolov8n-v1`, `yolov8n-v2`, `yolov8n-v2-int8`.

## 6. Setup & dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

- Core metrics/runners: **standard library only**.
- `pyyaml`: optional, for YAML config files (`eval_framework/core/config.py`).
- `anthropic`: optional, only needed for `eval_framework/llm/judge.py` (LLM-as-judge).
- `pytest`: for running `tests/`.

## 7. Running the benchmark again (step-by-step)

```bash
# 1. Install dependencies
pip install -r requirements.txt --break-system-packages

# 2. Run the CV benchmark on the bundled sample data
python scripts/run_cv_eval.py \
  --ground-truth data/cv_sample/ground_truth.json \
  --model-name mock-detector-v1 --dataset-name cv_sample --seed 42

# 3. Run the LLM benchmark on the bundled sample data
python scripts/run_llm_eval.py \
  --test-cases data/llm_sample/test_cases.json \
  --model-name mock-llm-v1 --dataset-name llm_sample --seed 42

# 4. Inspect results
cat results/history.jsonl

# 5. Compare two runs/model versions
python scripts/compare_models.py --models mock-detector-v1 mock-detector-v2

# 6. Run the test suite
python -m pytest tests/ -v
```

## 8. Sample results (from the bundled synthetic dataset)

These were produced by an actual run of the pipeline against the bundled
sample data/mock model (see `results/` for the raw JSON) — not hardcoded:

**CV (`mock-detector-v1` on `cv_sample`, seed 42):**
```
Precision: 87.5%
Recall: 77.78%
F1 Score: 82.35%
mAP: 77.28%
FPS: 44.05
Latency (mean): 22.7 ms
```

**LLM (`mock-llm-v1` on `llm_sample`, seed 42):**
```
Relevance: 0.40
Groundedness: 0.094
Hallucination rate: 1.0
Latency (mean): 109.5 ms
Token usage: 217 tokens (~$0.0016)
```
(Low LLM scores are expected — the mock generator returns a generic
templated sentence rather than a real grounded answer; the point is
demonstrating the pipeline runs and derives every number from the
evaluation, not that this particular mock model is good.)
