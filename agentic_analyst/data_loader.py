"""Utilities for loading CSV files and inferring lightweight metadata.

This module keeps the agent local-first by performing all parsing in-memory
with pandas.  It exposes a Dataset abstraction that stores the dataframe plus
schema information, summary statistics, and quick descriptions that can be used
by the NL orchestrator to reason about the available data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_SAMPLE_ROWS = 5


@dataclass
class ColumnSummary:
    """Lightweight statistics that help the agent talk about a column."""

    dtype: str
    example_values: List[str] = field(default_factory=list)
    stats: Dict[str, float | int | str] = field(default_factory=dict)


@dataclass
class Dataset:
    """Container for a dataframe plus derived metadata."""

    name: str
    path: Path
    df: pd.DataFrame
    schema: Dict[str, str]
    column_summaries: Dict[str, ColumnSummary]
    sample_rows: pd.DataFrame

    def describe(self, max_columns: int = 8) -> str:
        """Return a concise, human-readable dataset description."""
        col_parts = []
        for idx, (col, summary) in enumerate(self.column_summaries.items()):
            if idx >= max_columns:
                col_parts.append("...")
                break
            example = (
                f" e.g. {summary.example_values[0]}"
                if summary.example_values
                else ""
            )
            col_parts.append(f"{col} ({summary.dtype}{example})")
        return (
            f"{self.name}: {len(self.df)} rows x {len(self.df.columns)} cols | "
            + ", ".join(col_parts)
        )


def infer_schema(df: pd.DataFrame) -> Dict[str, str]:
    """Map dataframe dtypes to a simplified schema vocabulary."""
    schema = {}
    for col, dtype in df.dtypes.items():
        if pd.api.types.is_numeric_dtype(dtype):
            schema[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            schema[col] = "datetime"
        elif pd.api.types.is_bool_dtype(dtype):
            schema[col] = "boolean"
        else:
            schema[col] = "categorical"
    return schema


def build_column_summaries(
    df: pd.DataFrame, schema: Dict[str, str], sample_rows: int = DEFAULT_SAMPLE_ROWS
) -> Dict[str, ColumnSummary]:
    """Compute descriptive stats per column."""
    summaries: Dict[str, ColumnSummary] = {}
    for col, col_type in schema.items():
        series = df[col].dropna()
        examples = [str(val) for val in series.head(sample_rows).tolist()]
        stats: Dict[str, float | int | str] = {}
        if col_type == "numeric":
            stats = {
                "min": float(series.min()) if not series.empty else np.nan,
                "max": float(series.max()) if not series.empty else np.nan,
                "mean": float(series.mean()) if not series.empty else np.nan,
            }
        elif col_type == "categorical":
            stats = {
                "unique": int(series.nunique()),
                "top": str(series.mode().iloc[0]) if not series.empty else "",
            }
        elif col_type == "datetime":
            stats = {
                "min": str(series.min()) if not series.empty else "",
                "max": str(series.max()) if not series.empty else "",
            }
        summaries[col] = ColumnSummary(
            dtype=col_type, example_values=examples, stats=stats
        )
    return summaries


def load_csv(
    path: str | Path,
    *,
    name: Optional[str] = None,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    **read_csv_kwargs,
) -> Dataset:
    """Load a single CSV into a Dataset object."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if "sep" not in read_csv_kwargs:
        # Use Python's engine so pandas can sniff delimiters like ';'
        read_csv_kwargs.setdefault("engine", "python")
        read_csv_kwargs["sep"] = None
    df = pd.read_csv(csv_path, **read_csv_kwargs)
    schema = infer_schema(df)
    column_summaries = build_column_summaries(df, schema, sample_rows)
    sample = df.head(sample_rows)
    dataset_name = name or csv_path.stem
    return Dataset(
        name=dataset_name,
        path=csv_path,
        df=df,
        schema=schema,
        column_summaries=column_summaries,
        sample_rows=sample,
    )


def load_csvs(
    paths: Sequence[str | Path],
    *,
    names: Optional[Sequence[str]] = None,
    **read_csv_kwargs,
) -> Dict[str, Dataset]:
    """Load multiple CSVs and return a mapping of dataset name -> Dataset."""
    datasets: Dict[str, Dataset] = {}
    for idx, path in enumerate(paths):
        dataset_name = None
        if names and idx < len(names):
            dataset_name = names[idx]
        dataset = load_csv(path, name=dataset_name, **read_csv_kwargs)
        datasets[dataset.name] = dataset
    return datasets


def describe_dataset(dataset: Dataset) -> str:
    """Wrapper for dataset.describe to keep the module API consistent."""
    return dataset.describe()


def infer_join_candidates(
    datasets: Iterable[Dataset], *, min_unique_ratio: float = 0.05
) -> Dict[Tuple[str, str], List[str]]:
    """Suggest potential join keys between every pair of datasets.

    We treat a column as a viable key if it appears in both datasets with the
    same inferred dtype and has an adequate number of distinct values relative
    to the dataset size (to avoid joining on low-cardinality categories).
    """
    datasets_list = list(datasets)
    suggestions: Dict[Tuple[str, str], List[str]] = {}
    for i, left in enumerate(datasets_list):
        for right in datasets_list[i + 1 :]:
            pair_key = (left.name, right.name)
            candidates: List[str] = []
            for col, dtype in left.schema.items():
                if (
                    col in right.schema
                    and right.schema[col] == dtype
                    and col in left.df.columns
                    and col in right.df.columns
                ):
                    # skip extremely low-cardinality fields
                    cardinality = left.df[col].nunique(dropna=True)
                    if cardinality / max(len(left.df), 1) < min_unique_ratio:
                        continue
                    candidates.append(col)
            if candidates:
                suggestions[pair_key] = candidates
    return suggestions

