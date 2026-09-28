"""Benchmark an LLM on a json file of test cases.

    export ANTHROPIC_API_KEY=...
    python run_llm.py --name sonnet-run1 --model claude-sonnet-5 --price-in 3 --price-out 15

Prices are USD per 1M tokens, check the current price list for your model.
Without them, cost is reported as not applicable.
"""
import argparse
import json
from pathlib import Path

from evalkit.llm_eval import run
from evalkit.store import save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True, help="model id sent to the API")
    ap.add_argument("--cases", default="data/llm/cases.json")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--halluc-thr", type=float, default=0.6, help="groundedness below this = hallucinated")
    ap.add_argument("--price-in", type=float, default=None)
    ap.add_argument("--price-out", type=float, default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    from models.claude_api import make_generate  # imported here so the rest works without the sdk

    cases = json.loads(Path(a.cases).read_text())
    generate = make_generate(a.model, a.max_tokens, a.temperature)

    metrics, na, rows = run(cases, generate, a.halluc_thr, a.price_in, a.price_out)
    config = {k: v for k, v in vars(a).items() if k != "out"}
    path = save("llm", a.name, a.cases, metrics, config, na, out_dir=a.out)

    # keep the raw answers next to the summary so scores can be sanity-checked by eye
    (Path(a.out) / f"llm_{a.name}_answers.json").write_text(json.dumps(rows, indent=2))

    print(json.dumps({"metrics": metrics, "not_applicable": na}, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
