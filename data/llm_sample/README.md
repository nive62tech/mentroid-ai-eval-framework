# LLM Sample Test Set

A small **synthetic** QA test set (5 cases) used to demonstrate and
smoke-test the LLM evaluation pipeline end-to-end.

## Format

`test_cases.json` is a list of objects:

```json
{
  "query": "The question/prompt sent to the model",
  "context": "Optional supporting context/document the model should ground its answer in",
  "reference": "Optional expected/gold answer, used for accuracy scoring"
}
```

- `context` is optional. If omitted for all cases, groundedness and
  hallucination-rate are marked **not applicable** in the results (rather
  than silently defaulting to 0 or 1).
- `reference` is optional. If omitted for all cases, accuracy is marked
  **not applicable**.

## Using your own dataset

1. Write your real prompts/questions as `query`.
2. Where relevant, supply the `context` your RAG/system actually retrieves
   or is given, so groundedness reflects your real pipeline.
3. Where you have gold answers, supply `reference` for accuracy scoring.
4. Point `scripts/run_llm_eval.py --test-cases <path>` at your file.

### Requirements for a valid LLM test set
- At least 20-30 cases recommended for stable aggregate metrics; this
  5-case sample is for pipeline verification only.
- Cases should reflect the real distribution of queries the system will see
  in production (mix of easy/hard, in-scope/out-of-scope, etc.).
