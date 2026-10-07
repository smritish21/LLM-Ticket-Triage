"""Shared test helpers (pytest loads this file automatically)."""

import pytest


class FakeLLMClient:
    """A test double that replaces the real LLM.

    It returns pre-programmed replies in order and records every call, so
    tests can check both what the app did with the reply and what it sent.
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []  # list of (system_prompt, user_prompt)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self.replies:
            raise AssertionError("FakeLLMClient ran out of replies")
        return self.replies.pop(0)


@pytest.fixture
def make_fake_llm():
    """Usage in a test: fake = make_fake_llm(['reply 1', 'reply 2'])"""
    return FakeLLMClient
