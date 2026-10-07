"""Try the assistant from the command line.

Usage:
    python -m app "I was charged twice for my subscription"
"""

import sys

from app.llm_client import OpenAIClient
from app.triage import triage_ticket


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m app "<ticket text>"')
        sys.exit(1)

    result = triage_ticket(sys.argv[1], OpenAIClient())
    print(f"Category: {result.category}")
    print(f"Priority: {result.priority}")
    print(f"Summary:  {result.summary}")


if __name__ == "__main__":
    main()
