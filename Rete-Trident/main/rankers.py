"""Probability-table adapter for RETE's variable ranker.

The trained CodeBERT weights are not in this archive. This adapter consumes
probabilities exported by a trained ranker; it does not train or approximate one.
"""

import json
from pathlib import Path


class ChainRanker:
    def __init__(self, model=None):
        self.path = Path(model) if model else None
        self.table = None
        if self.path is not None:
            with self.path.open(encoding="utf-8") as stream:
                self.table = json.load(stream)
            if not isinstance(self.table, dict):
                raise ValueError("ranker probability file must be a JSON object")

    def has_models(self):
        return (self.table is not None
                and bool(self.table.get("default") or self.table.get("contexts")))

    def probabilities(self, template_code, hole_path, candidate_names):
        if not self.has_models():
            raise ValueError("a probability export is required")
        contexts = self.table.get("contexts", {})
        path_key = template_code + "@" + "/".join(hole_path)
        weights = contexts.get(path_key, contexts.get(template_code,
                                                        self.table.get("default", {})))
        if not isinstance(weights, dict):
            raise ValueError(f"invalid probabilities for {path_key}")
        if not weights:
            raise ValueError(f"no variable probabilities for {path_key}")
        result = {}
        for name in candidate_names:
            if name not in weights:
                continue
            probability = float(weights[name])
            if not 0 < probability <= 1:
                raise ValueError(f"probability for {name} must be in (0, 1]")
            result[name] = probability
        return result
