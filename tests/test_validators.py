"""Tests for parsing and validating the model's reply."""

import pytest

from app.validators import InvalidLLMOutput, TriageResult, extract_json, validate_result

VALID = {"category": "billing", "priority": "high", "summary": "Customer was charged twice."}


# ---------- extract_json ----------

@pytest.mark.parametrize(
    "raw_reply",
    [
        '{"category": "billing"}',                                  # plain JSON
        '```json\n{"category": "billing"}\n```',                    # markdown code fence
        'Sure! Here is the result: {"category": "billing"}',        # extra text before
        '{"category": "billing"}\nLet me know if you need more.',   # extra text after
    ],
    ids=["plain", "code-fence", "text-before", "text-after"],
)
def test_extract_json_handles_common_reply_formats(raw_reply):
    assert extract_json(raw_reply) == {"category": "billing"}


def test_extract_json_raises_when_no_json():
    with pytest.raises(InvalidLLMOutput, match="No JSON object"):
        extract_json("I'm sorry, I can't help with that.")


def test_extract_json_raises_on_broken_json():
    with pytest.raises(InvalidLLMOutput, match="not valid JSON"):
        extract_json('{"category": "billing",}')  # trailing comma


# ---------- validate_result ----------

def test_valid_data_returns_result():
    assert validate_result(VALID) == TriageResult(
        category="billing", priority="high", summary="Customer was charged twice."
    )


def test_values_are_normalised():
    data = {"category": " Billing ", "priority": "HIGH", "summary": "  Charged twice. "}
    result = validate_result(data)
    assert result.category == "billing"
    assert result.priority == "high"
    assert result.summary == "Charged twice."


@pytest.mark.parametrize("missing_field", ["category", "priority", "summary"])
def test_missing_field_raises(missing_field):
    data = {k: v for k, v in VALID.items() if k != missing_field}
    with pytest.raises(InvalidLLMOutput, match="Missing fields"):
        validate_result(data)


def test_unknown_category_raises():
    with pytest.raises(InvalidLLMOutput, match="Unknown category"):
        validate_result({**VALID, "category": "refunds"})


def test_unknown_priority_raises():
    with pytest.raises(InvalidLLMOutput, match="Unknown priority"):
        validate_result({**VALID, "priority": "urgent"})


@pytest.mark.parametrize("summary", ["", "   "])
def test_empty_summary_raises(summary):
    with pytest.raises(InvalidLLMOutput, match="Summary is empty"):
        validate_result({**VALID, "summary": summary})
