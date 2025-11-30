"""Session-level state management for the analyst agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

from .data_loader import Dataset, infer_join_candidates

Role = Literal["user", "assistant"]


@dataclass
class ArtifactMetadata:
    """Reference to any file or object produced during a turn."""

    artifact_type: str
    description: str
    path: Optional[Path] = None


@dataclass
class Turn:
    """Tracks a single conversational exchange."""

    role: Role
    message: str
    dataset_refs: Optional[List[str]] = None
    analysis_steps: Optional[List[str]] = None
    code: Optional[str] = None
    artifacts: List[ArtifactMetadata] = field(default_factory=list)
    insights: Optional[str] = None


@dataclass
class SessionConfig:
    """Configuration knobs for the session."""

    max_history_turns: int = 20
    output_dir: Path = Path("outputs")


@dataclass
class SessionState:
    """Holds datasets, chat history, and derived metadata."""

    datasets: Dict[str, Dataset]
    config: SessionConfig = field(default_factory=SessionConfig)
    chat_history: List[Turn] = field(default_factory=list)
    join_suggestions: Dict[tuple[str, str], List[str]] = field(default_factory=dict)
    sandbox_globals: Dict[str, object] = field(default_factory=dict)
    last_results: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.config.output_dir.mkdir(exist_ok=True, parents=True)
        if not self.join_suggestions:
            self.join_suggestions = infer_join_candidates(self.datasets.values())

    def append_turn(self, turn: Turn) -> None:
        """Add a new turn and trim history."""
        self.chat_history.append(turn)
        if len(self.chat_history) > self.config.max_history_turns:
            excess = len(self.chat_history) - self.config.max_history_turns
            self.chat_history = self.chat_history[excess:]

    def get_recent_history(self, limit: Optional[int] = None) -> List[Turn]:
        """Return the most recent turns."""
        limit = limit or self.config.max_history_turns
        return self.chat_history[-limit:]

    def get_dataset(self, name: str) -> Optional[Dataset]:
        """Retrieve a dataset by name, case-insensitive."""
        normalized = {key.lower(): key for key in self.datasets.keys()}
        key = normalized.get(name.lower())
        return self.datasets.get(key) if key else None

    def default_dataset(self) -> Optional[Dataset]:
        """Return a default dataset if none explicitly requested."""
        if self.datasets:
            first_key = next(iter(self.datasets.keys()))
            return self.datasets[first_key]
        return None

    def last_dataset_refs(self) -> List[str]:
        """Return dataset refs from the most recent turn if available."""
        for turn in reversed(self.chat_history):
            if turn.dataset_refs:
                return turn.dataset_refs
        return []

    def remember_results(self, label: str, obj: object) -> None:
        """Store a recent result for later transformations."""
        self.last_results[label] = obj


def initialize_session(
    datasets: Dict[str, Dataset], *, output_dir: Optional[str | Path] = None
) -> SessionState:
    """Factory helper to build a SessionState with sensible defaults."""
    config = SessionConfig(
        output_dir=Path(output_dir) if output_dir else SessionConfig().output_dir
    )
    return SessionState(datasets=datasets, config=config)

