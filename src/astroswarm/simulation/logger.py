
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class RunLogger:
    

    def __init__(self, run_name: str, config: dict, out_dir: str | Path = "results/runs"):
        self.run_name = run_name
        self.config = config
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save(self, metrics, extra: dict[str, Any] | None = None) -> Path:
        
        record = {
            "run_name": self.run_name,
            "seed": self.config.get("seed"),
            "config": self.config,
            "summary": metrics.summary(),
            "series": {k: v.tolist() for k, v in metrics.as_arrays().items()},
        }
        if extra:
            record["extra"] = extra
        path = self.out_dir / f"{self.run_name}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        return path
