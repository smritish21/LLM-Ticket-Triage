"""Main logic: send a ticket to the LLM and return a validated result."""

from app.llm_client import LLMClient
from app.prompts import SYSTEM_PROMPT, build_user_prompt
from app.validators import InvalidLLMOutput, TriageResult, extract_json, validate_result

MAX_ATTEMPTS = 2

# Used when the model keeps giving unusable answers, so a human looks at it.
FALLBACK_RESULT = TriageResult(
    category="other", priority="medium", summary="Needs manual review"
)


def triage_ticket(ticket_text: str, client: LLMClient) -> TriageResult:
    """Classify a support ticket.

    Retries once if the model's reply is malformed, then falls back to a
    safe default instead of crashing.
    """
    if not ticket_text or not ticket_text.strip():
        raise ValueError("Ticket text must not be empty")

    user_prompt = build_user_prompt(ticket_text)

    for _ in range(MAX_ATTEMPTS):
        raw_reply = client.complete(SYSTEM_PROMPT, user_prompt)
        try:
            return validate_result(extract_json(raw_reply))
        except InvalidLLMOutput:
            continue  # ask the model again

    return FALLBACK_RESULT
