"""
Generate lightweight textual previews of tabular data directories, used to
give the LLM agent context about the task's input data without dumping
full files into the prompt.
"""

from pathlib import Path

import pandas as pd


def preview_csv(p: Path) -> str:
    """Generate a short textual preview of a single csv file."""
    df = pd.read_csv(p)

    out = [f"-> {str(p)} has {df.shape[0]} rows and {df.shape[1]} columns."]

    cols = df.columns.tolist()
    out.append(f"The columns are: {', '.join(cols)}")

    # TODO(agent-improvement): summarize dtypes / missing values / target
    # column candidates here so the LLM gets richer feature-selection signal.

    return "\n".join(out)


def data_preview_generate(base_path: Path) -> str:
    """Generate a textual preview of every file in a data directory."""
    files = [p for p in Path(base_path).iterdir()]
    previews = [preview_csv(f) for f in sorted(files)]
    return "\n\n".join(previews)
