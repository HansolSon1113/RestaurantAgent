from __future__ import annotations

import re


def extract_json(content: str) -> str:
    """Return the JSON payload from an LLM response, stripping any markdown code fences.

    Many LLMs wrap JSON output in triple-backtick fences (e.g. ```json ... ```)
    even when explicitly instructed not to.  This function removes those fences
    so that ``json.loads`` can parse the result without errors.

    If no code fence is detected the original string is returned unchanged.
    """
    content = content.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if match:
        return match.group(1).strip()
    return content
