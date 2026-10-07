"""Turn the model's raw text reply into a checked TriageResult.

LLMs don't always follow the requested format, so we never trust the reply
blindly: we extract the JSON, then check every field.
"""

import json
import re
from dataclasses import dataclass

from app.prompts import CATEGORIES, PRIORITIES


class InvalidLLMOutput(ValueError):
    """Raised when the model's reply can't be turned into a valid result."""


@dataclass(frozen=True)
class TriageResult:
    category: str
    priority: str
    summary: str


def extract_json(raw: str) -> dict:
    """Find the JSON object in the reply.

    Models sometimes wrap JSON in ```json fences or add a sentence before it,
    so we look for the first '{' ... last '}' block instead of parsing the
    whole reply.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise InvalidLLMOutput(f"No JSON object found in reply: {raw[:80]!r}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise InvalidLLMOutput(f"Reply is not valid JSON: {exc}") from exc


def validate_result(data: dict) -> TriageResult:
    """Check required fields and allowed values, and normalise them."""
    missing = {"category", "priority", "summary"} - data.keys()
    if missing:
        raise InvalidLLMOutput(f"Missing fields: {sorted(missing)}")

    category = str(data["category"]).strip().lower()
    priority = str(data["priority"]).strip().lower()
    summary = str(data["summary"]).strip()

    if category not in CATEGORIES:
        raise InvalidLLMOutput(f"Unknown category: {category!r}")
    if priority not in PRIORITIES:
        raise InvalidLLMOutput(f"Unknown priority: {priority!r}")
    if not summary:
        raise InvalidLLMOutput("Summary is empty")

    return TriageResult(category=category, priority=priority, summary=summary)
