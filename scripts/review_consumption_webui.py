"""Web UI to validate and edit consumed-image annotations.

This tool reads the Label Studio JSON export and provides:
1. Validation of mask-vs-consumption consistency per task.
2. Overlay visualization of consumed image polygon masks.
3. Editing of numeric consumption fields (pct_*) and choice fields.
4. Save-back to a reviewed JSON file.

Run:
    python scripts/review_consumption_webui.py
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import unquote, urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_EXPORT_PATH = PROJECT_ROOT / "project-1-at-2026-07-21-18-54-66a41baf.json"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data_consumed" / "images"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data_consumed" / "project_consumption_reviewed.json"


# Metric aliases used by validation when metric field name differs from polygon label names.
METRIC_LABEL_ALIASES: Dict[str, List[str]] = {
    "pct_vanilla_pudding": ["vanilla_pudding_with_fruits"],
    "pct_salad_dish_main": ["salad_main"],
    # Composite dish metric: any component mask is acceptable.
    "pct_chicken_rice_veg": ["chicken", "rice", "carrots", "broccoli"],
}

CHOICE_DEFAULT_OPTIONS = ["Consumed", "Not consumed", "Not present"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review consumed-image annotations with a local web UI."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_EXPORT_PATH)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def extract_file_name_from_url(value: str) -> str:
    parsed = urlparse(value)
    return Path(unquote(parsed.path)).name


def suffix_stem(name: str) -> str:
    stem = Path(name).stem
    if "__" in stem:
        return stem.split("__", 1)[1]
    return stem


class ReviewSession:
    def __init__(self, export_path: Path, images_dir: Path, output_path: Path):
        self.export_path = export_path
        self.images_dir = images_dir
        self.output_path = output_path

        if not self.export_path.exists():
            raise FileNotFoundError(f"Input export not found: {self.export_path}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        with self.export_path.open("r", encoding="utf-8") as fh:
            self.tasks: List[Dict[str, Any]] = json.load(fh)

        self.image_path_by_suffix_stem = self._index_images(self.images_dir)

    @staticmethod
    def _index_images(images_dir: Path) -> Dict[str, Path]:
        indexed: Dict[str, Path] = {}
        for path in images_dir.iterdir():
            if path.is_file():
                indexed[suffix_stem(path.name)] = path
        return indexed

    def _get_task_image_info(self, task: Dict[str, Any]) -> Tuple[str, str, Path | None]:
        data = task.get("data", {})
        consumed_value = data.get("consumed")
        if isinstance(consumed_value, str) and consumed_value.startswith("http"):
            file_name = extract_file_name_from_url(consumed_value)
            stem = Path(file_name).stem
            return file_name, stem, self.image_path_by_suffix_stem.get(stem)

        for value in data.values():
            if isinstance(value, str) and value.startswith("http"):
                file_name = extract_file_name_from_url(value)
                stem = Path(file_name).stem
                return file_name, stem, self.image_path_by_suffix_stem.get(stem)

        return "", "", None

    @staticmethod
    def _find_annotation_results(task: Dict[str, Any]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for annotation in task.get("annotations", []):
            results.extend(annotation.get("result", []))
        return results

    @staticmethod
    def _expected_polygon_labels(metric_name: str) -> List[str]:
        if metric_name in METRIC_LABEL_ALIASES:
            return METRIC_LABEL_ALIASES[metric_name]
        if metric_name.startswith("pct_"):
            return [metric_name[4:]]
        return [metric_name]

    def validate_task(self, task: Dict[str, Any]) -> List[Dict[str, str]]:
        issues: List[Dict[str, str]] = []
        results = self._find_annotation_results(task)

        polygon_labels: List[str] = []
        numbers: Dict[str, List[float]] = {}
        choices: Dict[str, List[str]] = {}

        for result in results:
            result_type = result.get("type")
            from_name = result.get("from_name", "")
            value = result.get("value", {})

            if result_type == "polygonlabels":
                polygon_labels.extend(value.get("polygonlabels", []))
            elif result_type == "number":
                number_value = value.get("number")
                if isinstance(number_value, (int, float)):
                    numbers.setdefault(from_name, []).append(float(number_value))
            elif result_type == "choices":
                for entry in value.get("choices", []):
                    choices.setdefault(from_name, []).append(str(entry))

        # Rule 1: zero-consumption must be backed by matching mask label.
        for metric_name, metric_values in numbers.items():
            expected = self._expected_polygon_labels(metric_name)
            has_expected_mask = any(label in polygon_labels for label in expected)
            for metric_value in metric_values:
                if metric_value == 0 and not has_expected_mask:
                    issues.append(
                        {
                            "check": "zero_consumption_and_missing_mask",
                            "message": (
                                f"{metric_name}=0 but none of expected masks {expected} "
                                "was found in polygon labels."
                            ),
                        }
                    )

        # Rule 2: mutually exclusive states in the same choice field.
        for field_name, values in choices.items():
            normalized = {v.strip() for v in values}
            if "Consumed" in normalized and "Not consumed" in normalized:
                issues.append(
                    {
                        "check": "mutually_exclusive_state_validation",
                        "message": (
                            f"Field '{field_name}' contains both 'Consumed' and 'Not consumed'."
                        ),
                    }
                )

        # Rule 3: no local consumed image mapping found.
        _, _, image_path = self._get_task_image_info(task)
        if image_path is None:
            issues.append(
                {
                    "check": "image_mapping",
                    "message": "No local consumed image found for this task.",
                }
            )

        return issues

    def summarize_tasks(self) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for idx, task in enumerate(self.tasks):
            source_file, source_stem, image_path = self._get_task_image_info(task)
            issues = self.validate_task(task)
            summary.append(
                {
                    "idx": idx,
                    "task_id": task.get("id"),
                    "source_file": source_file,
                    "source_stem": source_stem,
                    "has_local_image": image_path is not None,
                    "issue_count": len(issues),
                }
            )
        return summary

    def task_detail(self, idx: int) -> Dict[str, Any]:
        task = self.tasks[idx]
        source_file, source_stem, image_path = self._get_task_image_info(task)
        results = self._find_annotation_results(task)

        polygons: List[Dict[str, Any]] = []
        numbers: List[Dict[str, Any]] = []
        choices: List[Dict[str, Any]] = []

        for result in results:
            result_type = result.get("type")
            from_name = result.get("from_name", "")
            value = result.get("value", {})

            if result_type == "polygonlabels":
                polygons.append(
                    {
                        "from_name": from_name,
                        "labels": value.get("polygonlabels", []),
                        "points": value.get("points", []),
                    }
                )
            elif result_type == "number":
                numbers.append(
                    {
                        "from_name": from_name,
                        "value": value.get("number"),
                    }
                )
            elif result_type == "choices":
                current_values = value.get("choices", [])
                options = list(CHOICE_DEFAULT_OPTIONS)
                for existing in current_values:
                    if existing not in options:
                        options.append(existing)
                choices.append(
                    {
                        "from_name": from_name,
                        "values": current_values,
                        "options": options,
                    }
                )

        return {
            "idx": idx,
            "task_id": task.get("id"),
            "source_file": source_file,
            "source_stem": source_stem,
            "image_url": f"/image/{idx}" if image_path is not None else None,
            "polygons": polygons,
            "numbers": numbers,
            "choices": choices,
            "issues": self.validate_task(task),
        }

    def update_task(self, idx: int, payload: Dict[str, Any]) -> None:
        task = self.tasks[idx]

        number_updates = payload.get("numbers", {})
        choice_updates = payload.get("choices", {})
        number_deletes = set(payload.get("delete_numbers", []))

        # Remove number entries completely when requested (for example pct_brownie).
        if number_deletes:
            for annotation in task.get("annotations", []):
                current_results = annotation.get("result", [])
                annotation["result"] = [
                    result
                    for result in current_results
                    if not (
                        result.get("type") == "number"
                        and result.get("from_name", "") in number_deletes
                    )
                ]

        results = self._find_annotation_results(task)

        for result in results:
            result_type = result.get("type")
            from_name = result.get("from_name", "")
            value = result.get("value", {})

            if result_type == "number" and from_name in number_updates:
                raw_val = number_updates[from_name]
                try:
                    value["number"] = float(raw_val)
                except (TypeError, ValueError):
                    continue

            if result_type == "choices" and from_name in choice_updates:
                new_val = choice_updates[from_name]
                if isinstance(new_val, list):
                    value["choices"] = [str(v) for v in new_val]
                elif isinstance(new_val, str):
                    value["choices"] = [new_val]

    def save(self) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as fh:
            json.dump(self.tasks, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        return self.output_path

    def image_path_for_idx(self, idx: int) -> Path | None:
        _, _, image_path = self._get_task_image_info(self.tasks[idx])
        return image_path


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Consumption Review UI</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; }
    .layout { display: grid; grid-template-columns: 340px 1fr; height: 100vh; }
    .left { border-right: 1px solid #ddd; overflow: auto; padding: 12px; }
    .right { padding: 12px; overflow: auto; }
    .task { padding: 8px; border: 1px solid #ddd; margin-bottom: 8px; cursor: pointer; }
    .task.bad { border-left: 4px solid #c62828; }
    .task.good { border-left: 4px solid #2e7d32; }
    .row { margin-bottom: 8px; }
    label { display: inline-block; min-width: 180px; }
    input, select { min-width: 180px; }
    .delete-toggle { margin-left: 10px; font-size: 12px; color: #444; min-width: 0; }
    .issues { background: #fff7f7; border: 1px solid #f5c2c2; padding: 10px; margin: 10px 0; }
    .toolbar { display: flex; gap: 8px; margin-bottom: 10px; }
    .canvas-wrap { position: relative; width: fit-content; }
    canvas { position: absolute; left: 0; top: 0; pointer-events: none; }
    img { max-width: 100%; display: block; }
    .small { color: #666; font-size: 12px; }
  </style>
</head>
<body>
  <div class=\"layout\">
    <div class="left">
      <div class="toolbar">
        <button onclick="refreshTasks()">Reload</button>
        <button onclick="saveAll()">Save JSON</button>
      </div>
      <div class="row" style="padding: 0 4px; font-size: 13px;">
        <label style="min-width: unset; cursor: pointer;">
          <input type="checkbox" id="showOnlyIssues" onchange="renderTaskList()" /> Show only tasks with issues
        </label>
      </div>
      <div id="taskList"></div>
    </div>
    <div class=\"right\">
      <h2 id=\"title\">Select a task</h2>
      <div id=\"meta\" class=\"small\"></div>
      <div id=\"issues\"></div>
      <div class=\"canvas-wrap\">
        <img id=\"img\" alt=\"consumed image\" />
        <canvas id=\"overlay\"></canvas>
      </div>
    <h3>Numbers</h3>
    <div class=\"small\">Tip: tick Delete to remove a numeric field entry from this task.</div>
      <div id=\"numbers\"></div>
      <h3>Choices</h3>
      <div id=\"choices\"></div>
      <div class=\"toolbar\">
        <button onclick=\"saveTask()\">Save Task Changes</button>
      </div>
    </div>
  </div>

  <script>
    let tasks = [];
    let current = null;

    async function api(url, options) {
      const res = await fetch(url, options);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    async function refreshTasks() {
      tasks = await api('/api/tasks');
      renderTaskList();
    }

    function renderTaskList() {
      const list = document.getElementById('taskList');
      list.innerHTML = '';
      const showOnlyIssues = document.getElementById('showOnlyIssues').checked;

      for (const t of tasks) {
        if (showOnlyIssues && t.issue_count <= 0) {
          continue;
        }
        const el = document.createElement('div');
        el.className = 'task ' + (t.issue_count > 0 ? 'bad' : 'good');
        el.innerHTML = `<b>#${t.idx}</b> task=${t.task_id}<br>${t.source_file || '(no file)'}<br>issues=${t.issue_count}`;
        el.onclick = () => loadTask(t.idx);
        list.appendChild(el);
      }
    }

    async function loadTask(idx) {
      current = await api('/api/task/' + idx);
      document.getElementById('title').textContent = `Task ${current.task_id} (#${current.idx})`;
      document.getElementById('meta').textContent = `source: ${current.source_file || '(missing)'}`;

      const issues = document.getElementById('issues');
      if (current.issues.length) {
        issues.className = 'issues';
        issues.innerHTML = '<b>Issues</b><br>' + current.issues.map(i => `${i.check}: ${i.message}`).join('<br>');
      } else {
        issues.className = '';
        issues.innerHTML = '<span class="small">No validation issues for this task.</span>';
      }

      renderNumbers(current.numbers);
      renderChoices(current.choices);
      renderImageAndPolygons(current);
    }

    function renderNumbers(numbers) {
      const wrap = document.getElementById('numbers');
      wrap.innerHTML = '';
      for (const n of numbers) {
        const row = document.createElement('div');
        row.className = 'row';
                row.innerHTML = `<label>${n.from_name}</label><input type=\"number\" step=\"1\" id=\"num_${n.from_name}\" value=\"${n.value ?? ''}\" /><label class=\"delete-toggle\"><input type=\"checkbox\" id=\"del_${n.from_name}\" /> Delete</label>`;
        wrap.appendChild(row);
      }
    }

    function renderChoices(choices) {
      const wrap = document.getElementById('choices');
      wrap.innerHTML = '';
      for (const c of choices) {
        const selected = (c.values && c.values.length) ? c.values[0] : '';
        const opts = c.options.map(o => `<option value=\"${o}\" ${o === selected ? 'selected' : ''}>${o}</option>`).join('');
        const row = document.createElement('div');
        row.className = 'row';
        row.innerHTML = `<label>${c.from_name}</label><select id=\"choice_${c.from_name}\">${opts}</select>`;
        wrap.appendChild(row);
      }
    }

    function drawPolygons(detail) {
      const img = document.getElementById('img');
      const canvas = document.getElementById('overlay');
      const ctx = canvas.getContext('2d');
      canvas.width = img.clientWidth;
      canvas.height = img.clientHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#ff3d00';
      ctx.fillStyle = 'rgba(255, 61, 0, 0.15)';

      for (const p of detail.polygons) {
        const points = p.points || [];
        if (!points.length) continue;
        ctx.beginPath();
        for (let i = 0; i < points.length; i++) {
          const x = (points[i][0] / 100) * canvas.width;
          const y = (points[i][1] / 100) * canvas.height;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
    }

    function renderImageAndPolygons(detail) {
      const img = document.getElementById('img');
      if (!detail.image_url) {
        img.removeAttribute('src');
        const canvas = document.getElementById('overlay');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        return;
      }

      img.onload = () => drawPolygons(detail);
      img.src = detail.image_url + '?t=' + Date.now();
    }

    async function saveTask() {
      if (!current) return;
      const numbers = {};
            const delete_numbers = [];
      for (const n of current.numbers) {
                const del = document.getElementById('del_' + n.from_name);
                if (del && del.checked) {
                    delete_numbers.push(n.from_name);
                    continue;
                }
        const el = document.getElementById('num_' + n.from_name);
        if (!el) continue;
        numbers[n.from_name] = el.value;
      }

      const choices = {};
      for (const c of current.choices) {
        const el = document.getElementById('choice_' + c.from_name);
        if (!el) continue;
        choices[c.from_name] = el.value;
      }

      await api('/api/task/' + current.idx, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({numbers, choices, delete_numbers}),
      });

      await loadTask(current.idx);
      await refreshTasks();
    }

    async function saveAll() {
      const res = await api('/api/save', { method: 'POST' });
      alert('Saved to: ' + res.path);
    }

    refreshTasks();
  </script>
</body>
</html>
"""


def build_handler(session: ReviewSession):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, text: str, status: int = 200) -> None:
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:
            if self.path == "/":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/api/tasks":
                self._send_json(session.summarize_tasks())
                return

            if self.path.startswith("/api/task/"):
                try:
                    idx = int(self.path.split("/")[-1])
                    self._send_json(session.task_detail(idx))
                except Exception as exc:
                    self._send_text(str(exc), status=400)
                return

            if self.path.startswith("/image/"):
                try:
                    idx_part = self.path.split("/")[-1].split("?")[0]
                    idx = int(idx_part)
                    path = session.image_path_for_idx(idx)
                    if path is None or not path.exists():
                        self._send_text("Image not found", status=404)
                        return
                    data = path.read_bytes()
                    ext = path.suffix.lower()
                    mime = "image/jpeg"
                    if ext == ".png":
                        mime = "image/png"
                    elif ext == ".webp":
                        mime = "image/webp"
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except Exception as exc:
                    self._send_text(str(exc), status=400)
                return

            self._send_text("Not found", status=404)

        def do_POST(self) -> None:
            if self.path.startswith("/api/task/"):
                try:
                    idx = int(self.path.split("/")[-1])
                    payload = self._read_json_body()
                    session.update_task(idx, payload)
                    self._send_json({"ok": True})
                except Exception as exc:
                    self._send_text(str(exc), status=400)
                return

            if self.path == "/api/save":
                try:
                    out_path = session.save()
                    self._send_json({"ok": True, "path": str(out_path)})
                except Exception as exc:
                    self._send_text(str(exc), status=500)
                return

            self._send_text("Not found", status=404)

        def log_message(self, format: str, *args: Any) -> None:
            # Keep terminal output focused on review progress.
            return

    return Handler


def main() -> int:
    args = parse_args()
    session = ReviewSession(
        export_path=args.input.resolve(),
        images_dir=args.images_dir.resolve(),
        output_path=args.output.resolve(),
    )

    handler = build_handler(session)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"Review UI running at http://{args.host}:{args.port}")
    print(f"Input export : {session.export_path}")
    print(f"Images dir   : {session.images_dir}")
    print(f"Save output  : {session.output_path}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
