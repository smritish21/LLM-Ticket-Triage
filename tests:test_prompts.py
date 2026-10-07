"""Tests for prompt building."""

import pytest

from app.prompts import (
    CATEGORIES,
    MAX_TICKET_CHARS,
    PRIORITIES,
    SYSTEM_PROMPT,
    build_user_prompt,
)


@pytest.mark.parametrize("value", CATEGORIES + PRIORITIES)
def test_system_prompt_lists_every_allowed_value(value):
    # If a category is added in code but not in the prompt, the model can never pick it.
    assert value in SYSTEM_PROMPT


def test_system_prompt_asks_for_json_only():
    assert "JSON" in SYSTEM_PROMPT
    assert "no other text" in SYSTEM_PROMPT


def test_user_prompt_contains_ticket_inside_delimiters():
    prompt = build_user_prompt("My order is late")
    assert "<<<\nMy order is late\n>>>" in prompt


def test_user_prompt_strips_surrounding_whitespace():
    prompt = build_user_prompt("   My order is late \n\n")
    assert "<<<\nMy order is late\n>>>" in prompt


def test_long_ticket_is_truncated():
    long_ticket = "a" * (MAX_TICKET_CHARS + 500)
    prompt = build_user_prompt(long_ticket)
    assert "[truncated]" in prompt
    assert "a" * (MAX_TICKET_CHARS + 1) not in prompt


def test_ticket_at_the_limit_is_not_truncated():
    # Boundary value: exactly MAX_TICKET_CHARS should pass through unchanged.
    prompt = build_user_prompt("a" * MAX_TICKET_CHARS)
    assert "[truncated]" not in prompt
