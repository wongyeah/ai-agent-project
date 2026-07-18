"""
Structured generation: get a validated Pydantic object back from an LLM
backend, instead of raw text.

This works with ANY backend that implements `LLMBackend.generate_response`
(see src/llm/backend.py) — it doesn't require the backend to support
native function calling / JSON mode. The approach:

    1. Describe the required JSON schema in the prompt.
    2. Call the backend as normal (plain text in, plain text out).
    3. Try to find + parse a JSON object in the response.
    4. Validate it against the Pydantic model.
    5. If parsing/validation fails, retry with the error message fed back
       to the model so it can correct itself.

If you switch to a hosted API that supports native structured outputs /
function calling (e.g. OpenAI's `response_format`, Anthropic's tool use),
you can replace this with a thinner wrapper that calls that native
mechanism directly instead of relying on prompting + retries. Both
approaches are worth knowing: this one is more portable (works with any
model), the native one is more reliable and doesn't waste retries.
"""

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from src.llm.backend import LLMBackend

T = TypeVar("T", bound=BaseModel)


def _extract_json_blob(text: str) -> str | None:
    """
    Pull out the first plausible JSON object from a text blob.

    Handles the common case of the model wrapping its JSON in a
    ```json ... ``` fence, or just emitting stray text around it.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)

    # Fallback: take everything between the first "{" and the last "}".
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return None


def generate_structured(
    llm: LLMBackend,
    system_message: str,
    user_message: str,
    response_model: type[T],
    retries: int = 3,
) -> T:
    """
    Ask the LLM to respond in JSON matching `response_model`, and return a
    validated instance of it. Raises ValueError if all retries fail.
    """
    schema_json = json.dumps(response_model.model_json_schema(), indent=2)

    format_instructions = (
        "\n\nRespond with ONLY a single JSON object (no other text, no "
        "markdown fences) matching this JSON schema:\n"
        f"{schema_json}"
    )

    current_user_message = user_message + format_instructions
    last_error: str | None = None

    for attempt in range(retries):
        if last_error is not None:
            current_user_message = (
                user_message
                + format_instructions
                + f"\n\nYour previous response was invalid: {last_error}\n"
                "Please try again and return ONLY valid JSON matching the schema."
            )

        raw_response = llm.generate_response(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": current_user_message},
            ]
        )

        json_blob = _extract_json_blob(raw_response)
        if json_blob is None:
            last_error = "No JSON object found in the response."
            continue

        try:
            parsed = json.loads(json_blob)
        except json.JSONDecodeError as e:
            last_error = f"Response was not valid JSON: {e}"
            continue

        try:
            return response_model.model_validate(parsed)
        except ValidationError as e:
            last_error = f"JSON did not match the required schema: {e}"
            continue

    raise ValueError(
        f"Failed to get a valid {response_model.__name__} after {retries} "
        f"attempts. Last error: {last_error}"
    )
