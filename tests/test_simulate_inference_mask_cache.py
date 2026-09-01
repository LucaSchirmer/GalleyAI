from __future__ import annotations

import json
from pathlib import Path

from scripts.simulate_inference import resolve_detections


def test_resolve_detections_uses_shared_cache_without_running_yolo(tmp_path: Path) -> None:
    image_path = tmp_path / "abc123__tray.jpg"
    shared_cache = tmp_path / "shared"
    run_cache = tmp_path / "run"
    shared_cache.mkdir()
    cached = [{"class": "rice", "polygon": [[0.0, 0.0]], "bbox": [0, 0, 1, 1]}]
    (shared_cache / "tray.json").write_text(
        json.dumps({"detections": cached}), encoding="utf-8"
    )

    def fail_if_called(_image_path: Path):
        raise AssertionError("YOLO ran despite a mask-cache hit")

    detections, source = resolve_detections(
        image_path,
        shared_cache,
        run_cache,
        segment=fail_if_called,
    )

    assert detections == cached
    assert source == "cache"
    assert json.loads((run_cache / "tray.json").read_text(encoding="utf-8")) == {
        "detections": cached
    }
