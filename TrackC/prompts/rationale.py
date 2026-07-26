
"""Cached one-shot reasoning traces for Track C.

For selected reasoning techniques, a one-shot example may include a cached
reasoning trace generated once by a strong reference model. The benchmark
models then reuse the same exemplar and trace, improving comparability.

Track C uses whole-document extractive summarization, so the task key should
normally be "summary" rather than ACLSum aspects such as challenge, approach,
or outcome.

Default file layout:

    data/shots/rationales/<dataset>/<technique>__summary.json

The active cache directory is configured at runtime by engine.runner through
configure_rationale_cache().
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


PathLike = Union[str, Path]

# Default fallback. engine.runner may replace this using the experiment YAML.
_RATIONALE_CACHE_DIR = Path("data/shots/rationales")

# Techniques whose one-shot demonstrations may include explicit reasoning.
REASONING_TECHNIQUES = {
    "chain_of_thought",
    "self_ask",
    "scoring_based",
    "salience_inference",
}


@dataclass
class RationaleShot:
    """One cached rationale demonstration."""

    technique: str
    aspect: str
    source_model: str
    exemplar_sentences: List[str]
    shown_indices: List[int]
    gold_indices: List[int]
    f1: float
    rationale: str


# Cache key:
# (absolute cache directory, technique, task/aspect)
_cache: Dict[
    Tuple[str, str, str],
    Optional[RationaleShot],
] = {}


def configure_rationale_cache(
    cache_dir: PathLike,
) -> None:
    """Set the directory used by load_rationale() and save_rationale().

    This should be called once by engine.runner after reading the experiment
    YAML. For example:

        data/shots/rationales/xsum
        data/shots/rationales/cnndm
    """
    global _RATIONALE_CACHE_DIR

    resolved = Path(cache_dir).expanduser()

    if resolved.exists() and not resolved.is_dir():
        raise ValueError(
            f"Rationale cache path is not a directory: {resolved}"
        )

    _RATIONALE_CACHE_DIR = resolved

    # Cached entries may belong to the previous configured directory.
    _cache.clear()


def get_rationale_cache_dir() -> Path:
    """Return the currently configured rationale-cache directory."""
    return _RATIONALE_CACHE_DIR


def _normalize_component(
    value: str,
    field_name: str,
) -> str:
    """Validate a filename component used for cached rationale files."""
    normalized = str(value or "").strip()

    if not normalized:
        raise ValueError(
            f"{field_name} cannot be empty."
        )

    # Prevent accidental directory traversal or nested paths.
    if any(token in normalized for token in ("/", "\\", "..")):
        raise ValueError(
            f"{field_name} contains invalid path characters: {normalized!r}"
        )

    return normalized


def _path(
    technique: str,
    aspect: str,
    base_dir: Path,
) -> Path:
    """Build the rationale JSON path."""
    safe_technique = _normalize_component(
        technique,
        "technique",
    )
    safe_aspect = _normalize_component(
        aspect,
        "aspect",
    )

    return base_dir / f"{safe_technique}__{safe_aspect}.json"


def _cache_key(
    base_dir: Path,
    technique: str,
    aspect: str,
) -> Tuple[str, str, str]:
    """Build a stable in-memory cache key."""
    return (
        str(base_dir.resolve()),
        str(technique),
        str(aspect),
    )


def _validate_shot(
    shot: RationaleShot,
    source_path: Optional[Path] = None,
) -> None:
    """Validate a loaded or newly created rationale shot."""
    location = (
        f" in {source_path}"
        if source_path is not None
        else ""
    )

    if not shot.technique.strip():
        raise ValueError(
            f"Rationale technique is empty{location}."
        )

    if not shot.aspect.strip():
        raise ValueError(
            f"Rationale task/aspect is empty{location}."
        )

    if not shot.source_model.strip():
        raise ValueError(
            f"Rationale source_model is empty{location}."
        )

    if not isinstance(shot.exemplar_sentences, list):
        raise ValueError(
            f"exemplar_sentences must be a list{location}."
        )

    if not shot.exemplar_sentences:
        raise ValueError(
            f"exemplar_sentences cannot be empty{location}."
        )

    if not isinstance(shot.shown_indices, list):
        raise ValueError(
            f"shown_indices must be a list{location}."
        )

    if not isinstance(shot.gold_indices, list):
        raise ValueError(
            f"gold_indices must be a list{location}."
        )

    n_sentences = len(shot.exemplar_sentences)

    for field_name, values in (
        ("shown_indices", shot.shown_indices),
        ("gold_indices", shot.gold_indices),
    ):
        seen = set()

        for value in values:
            try:
                index = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{field_name} contains a non-integer value "
                    f"{value!r}{location}."
                ) from exc

            if not 1 <= index <= n_sentences:
                raise ValueError(
                    f"{field_name} contains out-of-range index "
                    f"{index}{location}; document has "
                    f"{n_sentences} sentences."
                )

            if index in seen:
                raise ValueError(
                    f"{field_name} contains duplicate index "
                    f"{index}{location}."
                )

            seen.add(index)

    try:
        score = float(shot.f1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Rationale f1 must be numeric{location}."
        ) from exc

    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f"Rationale f1 must be between 0 and 1{location}, "
            f"got {score}."
        )

    if not str(shot.rationale or "").strip():
        raise ValueError(
            f"Rationale text is empty{location}."
        )


def load_rationale(
    technique: str,
    aspect: str,
    base_dir: Optional[PathLike] = None,
) -> Optional[RationaleShot]:
    """Load one cached rationale demonstration.

    Parameters
    ----------
    technique:
        Prompting technique name, for example ``chain_of_thought``.
    aspect:
        Compatibility task key. In Track C this should normally be ``summary``.
    base_dir:
        Optional explicit directory. When omitted, the runtime-configured
        rationale directory is used.

    Returns
    -------
    Optional[RationaleShot]
        The cached shot, or ``None`` when no file exists.
    """
    base = (
        Path(base_dir).expanduser()
        if base_dir is not None
        else _RATIONALE_CACHE_DIR
    )

    key = _cache_key(
        base,
        technique,
        aspect,
    )

    if key in _cache:
        return _cache[key]

    path = _path(
        technique,
        aspect,
        base,
    )

    if not path.exists():
        _cache[key] = None
        return None

    if not path.is_file():
        raise ValueError(
            f"Rationale path is not a file: {path}"
        )

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in rationale file: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"Rationale JSON must contain an object: {path}"
        )

    try:
        shot = RationaleShot(**payload)
    except TypeError as exc:
        raise ValueError(
            f"Rationale file has missing or unexpected fields: {path}"
        ) from exc

    _validate_shot(
        shot,
        source_path=path,
    )

    # Ensure the requested file matches its internal metadata.
    if shot.technique != technique:
        raise ValueError(
            f"Rationale technique mismatch in {path}: "
            f"requested {technique!r}, file contains "
            f"{shot.technique!r}."
        )

    if shot.aspect != aspect:
        raise ValueError(
            f"Rationale task/aspect mismatch in {path}: "
            f"requested {aspect!r}, file contains "
            f"{shot.aspect!r}."
        )

    _cache[key] = shot
    return shot


def save_rationale(
    shot: RationaleShot,
    base_dir: Optional[PathLike] = None,
) -> Path:
    """Validate and save one rationale demonstration."""
    _validate_shot(shot)

    base = (
        Path(base_dir).expanduser()
        if base_dir is not None
        else _RATIONALE_CACHE_DIR
    )

    base.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = _path(
        shot.technique,
        shot.aspect,
        base,
    )

    path.write_text(
        json.dumps(
            asdict(shot),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    key = _cache_key(
        base,
        shot.technique,
        shot.aspect,
    )

    # Store the new value immediately.
    _cache[key] = shot

    return path


def clear_rationale_cache() -> None:
    """Clear only the in-memory rationale cache."""
    _cache.clear()

