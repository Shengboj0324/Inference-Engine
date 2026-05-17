"""Labelled-scenario discovery, validation, and iteration.

A scenario lives in ``data/scenarios/<scenario_id>/`` and consists of:

- ``observations.jsonl`` — one ``Observation`` per line (PII-scrubbed)
- ``gold_report.json``   — the labelled ``SituationReport`` for this scenario
- ``metadata.yaml``      — authorship, guideline version, split, PII sign-off

``ScenarioLoader.discover()`` walks the scenarios root and yields fully
validated ``ScenarioCase`` objects.  Invalid scenarios raise immediately so
malformed data cannot silently enter the training set or the judge run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterator, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.intelligence.situation_report import SituationReport


_VALID_SPLITS = frozenset({"train", "val", "heldout"})


class Observation(BaseModel):
    """A single source post / message in a scenario.

    ``observation_id`` is referenced by ``Citation.post_id`` in the gold
    report; ``Citation.char_start``/``char_end`` index into ``text``.
    """

    observation_id: str = Field(..., min_length=1, max_length=128)
    text: str = Field(..., min_length=1, max_length=20_000)
    source: str = Field(..., min_length=1, max_length=64)
    timestamp: Optional[str] = Field(None, max_length=64)

    model_config = ConfigDict(extra="forbid")


class ScenarioMetadata(BaseModel):
    """Provenance and routing metadata for a scenario."""

    scenario_id: str = Field(..., min_length=1, max_length=128)
    split: str
    author_id: str = Field(..., min_length=1, max_length=64)
    annotator_ids: List[str] = Field(..., min_length=1)
    adjudicator_id: Optional[str] = Field(None, max_length=64)
    guideline_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    created_on: date
    adversarial_tags: List[str] = Field(default_factory=list)
    pii_review_passed: bool
    pii_reviewer_id: Optional[str] = Field(None, max_length=64)

    model_config = ConfigDict(extra="forbid")

    @field_validator("split")
    @classmethod
    def _validate_split(cls, v: str) -> str:
        if v not in _VALID_SPLITS:
            raise ValueError(
                f"split must be one of {sorted(_VALID_SPLITS)}, got {v!r}"
            )
        return v


@dataclass(frozen=True)
class ScenarioCase:
    """Fully validated scenario ready for training, eval, or judging."""

    scenario_id: str
    metadata: ScenarioMetadata
    observations: List[Observation]
    gold_report: SituationReport
    path: Path


class ScenarioLoader:
    """Discovers and validates labelled scenarios on disk."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        if not self._root.exists():
            raise FileNotFoundError(
                f"scenarios root does not exist: {self._root}"
            )

    def discover(
        self,
        *,
        split: Optional[str] = None,
        require_pii_signoff: bool = True,
    ) -> Iterator[ScenarioCase]:
        """Yield every valid scenario, optionally filtered by split.

        Args:
            split: If set, only yield scenarios whose metadata.split matches.
            require_pii_signoff: When True (default), scenarios without
                ``pii_review_passed=True`` are skipped (and logged).  Set to
                False only for local development.
        """
        if split is not None and split not in _VALID_SPLITS:
            raise ValueError(
                f"split must be one of {sorted(_VALID_SPLITS)}, got {split!r}"
            )

        for child in sorted(self._root.iterdir()):
            if not child.is_dir():
                continue
            if child.name.startswith("_") or child.name.startswith("."):
                continue
            case = self._load_one(child)
            if split is not None and case.metadata.split != split:
                continue
            if require_pii_signoff and not case.metadata.pii_review_passed:
                continue
            yield case

    def _load_one(self, folder: Path) -> ScenarioCase:
        obs_path = folder / "observations.jsonl"
        gold_path = folder / "gold_report.json"
        meta_path = folder / "metadata.yaml"
        for p in (obs_path, gold_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(f"missing required file: {p}")

        observations: List[Observation] = []
        with obs_path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    observations.append(Observation.model_validate_json(raw))
                except Exception as exc:
                    raise ValueError(
                        f"{obs_path}:{lineno} invalid observation: {exc}"
                    ) from exc
        if not observations:
            raise ValueError(f"{obs_path} contains no observations")

        with gold_path.open("r", encoding="utf-8") as fh:
            gold = SituationReport.model_validate(json.load(fh))

        with meta_path.open("r", encoding="utf-8") as fh:
            meta = ScenarioMetadata.model_validate(yaml.safe_load(fh))

        if meta.scenario_id != folder.name:
            raise ValueError(
                f"{meta_path}: scenario_id {meta.scenario_id!r} does not "
                f"match folder name {folder.name!r}"
            )

        obs_ids = {o.observation_id for o in observations}
        for i, cit in enumerate(gold.citations):
            if cit.post_id not in obs_ids:
                raise ValueError(
                    f"{gold_path}: citation {i} references unknown "
                    f"post_id {cit.post_id!r}"
                )

        return ScenarioCase(
            scenario_id=meta.scenario_id,
            metadata=meta,
            observations=observations,
            gold_report=gold,
            path=folder,
        )
