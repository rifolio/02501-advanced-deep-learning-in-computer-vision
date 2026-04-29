"""Load and validate persistent eval split manifests (JSON)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalSplitManifest:
    """Persisted eval protocol: which val images and (optionally) which categories to score."""

    version: int
    ann_file: str
    image_ids: list[int]
    # If set, COCOeval uses these catIds; dataset filters targets to these ids only.
    eval_cat_ids: list[int] | None
    seed: int | None = None
    description: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": self.version,
            "ann_file": self.ann_file,
            "image_ids": self.image_ids,
            "eval_cat_ids": self.eval_cat_ids,
        }
        if self.seed is not None:
            d["seed"] = self.seed
        if self.description:
            d["description"] = self.description
        return d


def load_eval_split(path: str | Path) -> EvalSplitManifest:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    version = int(raw["version"])
    if version != 1:
        raise ValueError(f"Unsupported manifest version: {version}")
    eval_cat = raw.get("eval_cat_ids")
    if eval_cat is not None:
        eval_cat = [int(x) for x in eval_cat]
    return EvalSplitManifest(
        version=version,
        ann_file=str(raw["ann_file"]),
        image_ids=[int(x) for x in raw["image_ids"]],
        eval_cat_ids=eval_cat,
        seed=raw.get("seed"),
        description=str(raw.get("description", "")),
    )


def save_eval_split(manifest: EvalSplitManifest, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_json_dict(), f, indent=2)
        f.write("\n")
