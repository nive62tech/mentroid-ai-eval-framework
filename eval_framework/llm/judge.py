"""
Optional LLM-as-judge scorer.

The default metrics in `llm_metrics.py` are lexical-overlap proxies that
work with zero setup. For higher-quality Accuracy / Relevance /
Groundedness scoring, plug in a real judge model here (e.g. an
Anthropic API call) — the rest of the framework only depends on this
module returning floats in [0, 1], so swapping it in is a one-line
change in the runner.

This file intentionally ships as a template: it does NOT hardcode any
API key. Set ANTHROPIC_API_KEY in your environment to use it.
"""

from __future__ import annotations

import json
import os
from typing import Optional

try:
    import anthropic  # type: ignore
    _HAS_SDK = True
except ImportError:
    _HAS_SDK = False


JUDGE_PROMPT_TEMPLATE = """You are an evaluation judge. Score the RESPONSE against the \
QUERY and CONTEXT on three dimensions, each from 0.0 to 1.0:

- accuracy: does the response correctly answer the query (compare to REFERENCE if given)?
- relevance: does the response actually address what was asked?
- groundedness: is every claim in the response supported by the CONTEXT (1.0 = fully \
supported, 0.0 = fabricated / unsupported)?

QUERY: {query}
CONTEXT: {context}
REFERENCE (optional): {reference}
RESPONSE: {response}

Respond ONLY with JSON: {{"accuracy": <float>, "relevance": <float>, "groundedness": <float>}}
"""


class LLMJudge:
    """Thin wrapper around an Anthropic model used purely as a scorer."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: Optional[str] = None):
        if not _HAS_SDK:
            raise ImportError(
                "The 'anthropic' package is required for LLMJudge. "
                "Install with: pip install anthropic --break-system-packages"
            )
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def score(self, query: str, response: str, context: str = "", reference: str = "") -> dict:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            query=query, context=context or "N/A", reference=reference or "N/A", response=response
        )
        message = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in message.content if hasattr(block, "text"))
        text = text.strip().strip("```json").strip("```").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"accuracy": None, "relevance": None, "groundedness": None, "raw": text}
