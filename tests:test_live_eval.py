"""Live evaluation against the real model, using a labelled ("golden") dataset.

These tests call the OpenAI API, so they are skipped unless OPENAI_API_KEY
is set. Run only these with:  pytest -m live

LLM answers can vary, so instead of demanding every single case is right,
we check that overall accuracy stays above a threshold. If a prompt change
makes accuracy drop, these tests catch the regression.
"""

import json
import os
from pathlib import Path

import pytest

from app.triage import FALLBACK_RESULT, triage_ticket

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"),
]

GOLDEN_FILE = Path(__file__).parent / "data" / "golden_tickets.json"
CATEGORY_ACCURACY_THRESHOLD = 0.8
HIGH_PRIORITY_RECALL_THRESHOLD = 0.8


@pytest.fixture(scope="module")
def live_results():
    """Run every golden ticket through the real model once and reuse the results."""
    from app.llm_client import OpenAIClient

    client = OpenAIClient()
    cases = json.loads(GOLDEN_FILE.read_text())
    return [(case, triage_ticket(case["ticket"], client)) for case in cases]


def test_category_accuracy_above_threshold(live_results):
    wrong = [
        f"{case['id']}: expected {case['expected_category']}, got {result.category}"
        for case, result in live_results
        if result.category != case["expected_category"]
    ]
    accuracy = 1 - len(wrong) / len(live_results)
    assert accuracy >= CATEGORY_ACCURACY_THRESHOLD, (
        f"Accuracy {accuracy:.0%} is below {CATEGORY_ACCURACY_THRESHOLD:.0%}. Wrong:\n"
        + "\n".join(wrong)
    )


def test_high_priority_tickets_are_not_missed(live_results):
    # Missing an urgent ticket is worse than over-flagging a normal one,
    # so we measure recall on the "high" cases specifically.
    high_cases = [(c, r) for c, r in live_results if c["expected_priority"] == "high"]
    found = [c for c, r in high_cases if r.priority == "high"]
    recall = len(found) / len(high_cases)
    assert recall >= HIGH_PRIORITY_RECALL_THRESHOLD, f"High-priority recall is only {recall:.0%}"


def test_model_never_needs_the_fallback(live_results):
    # The fallback means the model gave two unusable replies in a row.
    failed = [case["id"] for case, result in live_results if result == FALLBACK_RESULT]
    assert not failed, f"Model output could not be parsed for: {failed}"


def test_summaries_are_short(live_results):
    too_long = [case["id"] for case, result in live_results if len(result.summary) > 200]
    assert not too_long, f"Summaries over 200 characters: {too_long}"
