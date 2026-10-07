"""LLM client abstraction.

The rest of the app only depends on the `LLMClient` interface (one method:
`complete`). That makes the app easy to test: tests pass in a fake client
that returns fixed replies, so they run fast, offline and deterministically.
"""

import os
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send the prompts to a model and return its raw text reply."""
        ...


class OpenAIClient:
    """Real client that calls the OpenAI API.

    Needs the OPENAI_API_KEY environment variable. The model can be changed
    with the OPENAI_MODEL environment variable.
    """

    def __init__(self, model: str | None = None):
        # Imported here so the offline tests don't need the openai package.
        from openai import OpenAI

        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,  # make answers as repeatable as possible
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
