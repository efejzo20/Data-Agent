"""Restricted execution environment for generated code."""

from __future__ import annotations

import builtins
import contextlib
import io
import signal
from dataclasses import dataclass, field
from typing import Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from .session_state import SessionState

SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "sorted": sorted,
    "round": round,
    "__import__": __import__,
    "print": print,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "type": type,
    "isinstance": isinstance,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
}


@dataclass
class SandboxResult:
    success: bool
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: Dict[str, Figure] = field(default_factory=dict)
    insight: Optional[str] = None
    stdout: str = ""
    error: Optional[str] = None

    @property
    def has_artifacts(self) -> bool:
        return bool(self.tables or self.figures)


@contextlib.contextmanager
def _time_limit(seconds: int):
    def handler(signum, frame):
        raise TimeoutError("Sandbox execution timed out.")

    original = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original)


def execute_code(
    code: str,
    session: SessionState,
    description: str = "Code execution",
    timeout_seconds: int = 30,
) -> SandboxResult:
    """Execute raw Python code string in sandbox (for tool-based execution)."""
    datasets_env = {
        name: dataset.df.copy(deep=True) for name, dataset in session.datasets.items()
    }
    safe_globals = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "plt": plt,
        "datasets": datasets_env,
    }
    local_env: Dict[str, object] = {}
    stdout_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buffer):
            with _time_limit(timeout_seconds):
                exec(code, safe_globals, local_env)
    except Exception as exc:  # pylint: disable=broad-except
        return SandboxResult(
            success=False,
            error=str(exc),
            stdout=stdout_buffer.getvalue(),
            tables={},
            figures={},
        )

    # Auto-detect common output variable names
    tables = {}
    figures = {}
    
    # Look for DataFrames
    for var_name, value in local_env.items():
        if isinstance(value, pd.DataFrame) and not var_name.startswith("_"):
            tables[var_name] = value
    
    # Look for figures
    if "fig" in local_env and local_env["fig"] is not None:
        figures["fig"] = local_env["fig"]
    
    # Check for any Figure objects
    for var_name, value in local_env.items():
        if isinstance(value, Figure) and not var_name.startswith("_"):
            figures[var_name] = value

    insight = local_env.get("insight")
    return SandboxResult(
        success=True,
        tables=tables,
        figures=figures,
        insight=insight,
        stdout=stdout_buffer.getvalue(),
    )

