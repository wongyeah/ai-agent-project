"""
Structured schemas for LLM outputs that the agent needs to parse reliably.

Why this file exists: earlier versions of this project tried to extract
information (is this code buggy? what's the metric?) from free-form LLM
text using regex, which is brittle. Instead, we define an explicit schema
here and ask the LLM to fill it in as JSON, then validate with Pydantic.
"""

from pydantic import BaseModel, Field


class ExecutionEvaluation(BaseModel):
    """
    The agent's judgement about one execution attempt.

    This is what we ask the LLM to produce after showing it the code and
    its execution output (stdout/stderr/traceback).
    """

    is_buggy: bool = Field(
        description=(
            "True if the code raised an exception, produced no valid "
            "submission file, or otherwise failed to accomplish the task."
        )
    )
    metric: float | None = Field(
        default=None,
        description=(
            "The validation metric (Mean Squared Error) achieved by this "
            "solution, if it ran successfully and a metric could be "
            "determined. Null if the code is buggy or no metric is "
            "available."
        ),
    )
    summary: str = Field(
        description=(
            "A short (1-3 sentence) explanation of what happened: what the "
            "code did, whether it succeeded, and why the metric/bug "
            "judgement was made."
        )
    )
