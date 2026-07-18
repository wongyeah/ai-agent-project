"""
Text processing utilities for parsing LLM outputs.

Handles extraction of Python code blocks and JSON objects from raw LLM
completions, plus small helpers for wrapping/truncating text for prompts.
"""

import json
import re


def wrap_code(code: str, lang: str = "python") -> str:
    """Wrap code with triple backticks for inclusion in a prompt."""
    return f"```{lang}\n{code}\n```"


def is_valid_python_script(script: str) -> bool:
    """Check whether a string compiles as a valid Python script."""
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def extract_jsons(text: str) -> list[dict]:
    """
    Extract all top-level JSON objects from free-form text.

    Caveat: this uses a non-greedy regex and cannot handle nested JSON
    objects correctly. Prefer structured-output APIs (e.g. function
    calling / Pydantic schemas) over this when possible.
    """
    json_objects = []
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for match in matches:
        try:
            json_objects.append(json.loads(match))
        except json.JSONDecodeError:
            pass
    return json_objects


def trim_long_string(string: str, threshold: int = 5100, k: int = 2500) -> str:
    """Truncate a long string, keeping the first and last k characters."""
    if len(string) > threshold:
        first_k_chars = string[:k]
        last_k_chars = string[-k:]
        truncated_len = len(string) - 2 * k
        return (
            f"{first_k_chars}\n ... [{truncated_len} characters truncated] ... "
            f"\n{last_k_chars}"
        )
    return string


def extract_code(text: str) -> str:
    """
    Extract Python code blocks from LLM output text.

    Looks for fenced ```python ... ``` blocks first. If none are found,
    falls back to treating the entire text as code. Only blocks that
    compile as valid Python are kept.
    """
    parsed_codes = []

    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        parsed_codes.append(match[1])

    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            parsed_codes.append(matches[0][2])

    valid_code_blocks = [c for c in parsed_codes if is_valid_python_script(c)]
    return "\n\n".join(valid_code_blocks)


def extract_text_up_to_code(s: str) -> str:
    """Extract the natural-language plan text preceding the first code block."""
    if "```" not in s:
        return ""
    return s[: s.find("```")].strip()
