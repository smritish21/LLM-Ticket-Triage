# LLM Ticket Triage — Testing an LLM Application with pytest

A small customer-support assistant that uses an LLM to classify tickets, and a test suite showing how to test an LLM-powered app reliably.

Given a ticket like *"I was charged twice for my subscription"*, the assistant returns:

```
Category: billing
Priority: high
Summary:  Customer was charged twice and wants a refund.
```

## Why this project

LLM apps are hard to test because the model's answers aren't fully predictable. This project splits testing into two layers:

| Layer | What it checks | Calls the real model? | Runs in CI? |
|---|---|---|---|
| **Deterministic tests** | Prompt building, JSON parsing, validation, retry and fallback logic | No — uses a fake LLM | Yes |
| **Live evaluation** | How well the real model classifies a labelled ("golden") dataset | Yes | No (needs API key) |

The deterministic tests check *our code*. The live evaluation checks *the model + prompt*, using accuracy thresholds instead of exact matches, because answers can vary between runs.

## Project structure

```
app/
  llm_client.py     LLMClient interface + real OpenAI client
  prompts.py        System prompt, allowed categories/priorities, user prompt builder
  validators.py     Extracts JSON from the reply and validates every field
  triage.py         Main flow: call model -> validate -> retry once -> fallback
tests/
  conftest.py       FakeLLMClient test double
  test_prompts.py   Prompt content, delimiters, truncation, boundary values
  test_validators.py  Reply formats, broken JSON, missing/invalid fields
  test_triage.py    Happy path, retry, fallback, empty input, prompt injection
  test_live_eval.py   Accuracy and recall thresholds on the golden dataset
  data/golden_tickets.json   12 labelled tickets
.github/workflows/tests.yml   Runs the tests with coverage on every push
```

## What is tested

- **Prompt checks** — every allowed category/priority appears in the prompt; ticket text is wrapped in delimiters; long tickets are truncated (including the exact boundary).
- **Output parsing** — handles plain JSON, markdown code fences and extra text around the JSON; rejects broken JSON.
- **Validation** — missing fields, unknown categories/priorities, empty summaries; values are normalised (case, whitespace).
- **Error handling** — retries once on a bad reply, then falls back to "needs manual review" instead of crashing; empty tickets are rejected *before* calling the API.
- **Prompt injection** — ticket text containing instructions stays inside the data delimiters.
- **Live evaluation** — category accuracy ≥ 80%, high-priority recall ≥ 80%, no unparseable outputs, short summaries.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Deterministic tests (no API key needed)
pytest --cov=app --cov-report=term-missing

# Live evaluation against the real model
export OPENAI_API_KEY=sk-...     # Windows: set OPENAI_API_KEY=sk-...
pytest -m live

# Try the assistant yourself
python -m app "I was charged twice for my subscription"
```

The model defaults to `gpt-4o-mini`; set `OPENAI_MODEL` to use a different one.

## Possible next steps

- Grow the golden dataset and track accuracy over time
- Add a second model provider and compare results
- Use an LLM-evaluation library (e.g. DeepEval or promptfoo) to score summary quality
