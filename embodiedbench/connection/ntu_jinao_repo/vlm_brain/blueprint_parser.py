"""Utilities for extracting skill_blueprint JSON from VLM text."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


@dataclass
class ParseResult:
    ok: bool
    blueprint: dict[str, Any] | None = None
    error: str | None = None
    extracted_text: str | None = None


def extract_json_from_text(raw_text: str) -> ParseResult:
    """Extract and parse the first JSON object from VLM output."""

    if not raw_text or not raw_text.strip():
        return ParseResult(ok=False, error="empty_vlm_output")

    candidates = _json_code_blocks(raw_text)
    candidates.append(_first_balanced_object(raw_text))
    last_error = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = f"json_decode_error: {exc}"
            continue
        if not isinstance(parsed, dict):
            return ParseResult(ok=False, error="parsed_json_is_not_object", extracted_text=candidate)
        return ParseResult(ok=True, blueprint=parsed, extracted_text=candidate)

    return ParseResult(ok=False, error=last_error or "no_json_object_found")


def _json_code_blocks(text: str) -> list[str]:
    pattern = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
    return [match.group(1).strip() for match in pattern.finditer(text)]


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None
