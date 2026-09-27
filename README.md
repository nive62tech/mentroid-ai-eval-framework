# AI Model Evaluation & Benchmarking Framework

**Task 09 — VISIONSTRA / Mentroid**

A reusable, modular framework for evaluating and benchmarking AI/ML
models — both Computer Vision (detection) models and LLM / AI system
outputs — using measurable, reproducible metrics instead of qualitative
observation.

## What it does

- **CV metrics**: Precision, Recall, F1 Score, mAP, FPS, Inference latency
- **LLM metrics**: Accuracy, Relevance, Groundedness, Hallucination rate, Latency, Token usage / cost
- Computes results from **actual evaluation runs** against a dataset — nothing is hardcoded
- Saves every run's results to disk (JSON) and appends to a history log for comparing model versions
- Gracefully marks a metric as "not applicable" (with a reason) instead of guessing, e.g. when no reference answers or context are supplied
- Ships with tiny synthetic sample datasets + mock models so the whole pipeline can be run and verified with zero external setup

## Project layout

```
eval-framework/
├── eval_framework/          # the library
│   ├── core/                 # timing, result storage/comparison, config loading
│   ├── cv/                   # detection metrics + CV benchmark runner
│   └── llm/                  # LLM metrics + LLM benchmark runner + optional LLM-judge
├── scripts/                  # CLI entry points
│   ├── run_cv_eval.py
│   ├── run_llm_eval.py
│   └── compare_models.py
├── data/
│   ├── cv_sample/             # synthetic CV ground truth + dataset requirements doc
│   └── llm_sample/            # synthetic LLM test cases + dataset requirements doc
├── configs/                   # example YAML configs for CV / LLM runs
├── tests/                      # pytest unit tests (24 tests)
├── results/                    # evaluation outputs land here (gitignored contents, dir kept)
└── docs/METHODOLOGY.md         # evaluation methodology + reproduction instructions
```

## Setup

```bash
pip install -r requirements.txt --break-system-packages
```

The core framework has **zero required dependencies** beyond the Python
standard library. `pyyaml` and `anthropic` are only needed for optional
features (YAML configs, LLM-as-judge scoring).

## Quick start

### Run a CV benchmark
```bash
python scripts/run_cv_eval.py \
  --ground-truth data/cv_sample/ground_truth.json \
  --model-name mock-detector-v1 \
  --dataset-name cv_sample
```

### Run an LLM benchmark
```bash
python scripts/run_llm_eval.py \
  --test-cases data/llm_sample/test_cases.json \
  --model-name mock-llm-v1 \
  --dataset-name llm_sample
```

### Compare model versions
```bash
python scripts/compare_models.py --models mock-detector-v1 mock-detector-v2
```

### Run tests
```bash
python -m pytest tests/ -v
```

## Plugging in a real model

The bundled scripts use small **mock models** purely so the pipeline runs
out of the box with no API keys or GPU. To evaluate a real model, you
only need to supply a function — the framework handles metrics, timing,
FPS, saving, and comparison around it.

**CV:**
```python
from eval_framework.cv.runner import run_cv_benchmark

def predict_fn(image):
    # call your real model here
    return [{"class": "car", "bbox": [x1, y1, x2, y2], "score": 0.91}, ...]

result = run_cv_benchmark(
    predict_fn=predict_fn,
    images=your_images_dict,
    ground_truth=your_ground_truth_dict,
    model_name="yolov8n-v3",
    dataset_name="traffic_cam_v2",
)
```

**LLM:**
```python
from eval_framework.llm.runner import run_llm_benchmark

def generate_fn(prompt):
    response = your_model.generate(prompt)
    return {"text": response.text, "input_tokens": response.usage.input, "output_tokens": response.usage.output}

result = run_llm_benchmark(
    generate_fn=generate_fn,
    test_cases=your_test_cases,   # [{"query", "context", "reference"}, ...]
    model_name="claude-sonnet-4-6",
    dataset_name="support_qa_v1",
)
```

See `docs/METHODOLOGY.md` for full details on each metric, dataset
requirements, and how to reproduce results.


