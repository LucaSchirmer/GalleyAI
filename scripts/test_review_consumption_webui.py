from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("review_consumption_webui.py")
SPEC = importlib.util.spec_from_file_location("review_consumption_webui", MODULE_PATH)
assert SPEC and SPEC.loader
review_ui = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review_ui)


class ReviewSessionTest(unittest.TestCase):
    def build_session(self, records):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        after_dir = root / "after"
        before_dir = root / "before"
        pairs_dir = root / "pairs"
        masks_dir = root / "masks"
        for directory in (after_dir, before_dir, pairs_dir, masks_dir):
            directory.mkdir()

        before_path = before_dir / "abc__before.jpg"
        after_path = after_dir / "def__after.jpg"
        before_path.write_bytes(b"before")
        after_path.write_bytes(b"after")

        payload = {
            "task_count": len(records),
            "indexed_by_image_stem": {"after": records},
        }
        index_path = root / "index.json"
        output_path = root / "output.json"
        index_path.write_text(json.dumps(payload), encoding="utf-8")
        (pairs_dir / "train.json").write_text(
            json.dumps(
                [
                    {
                        "before": str(before_path),
                        "after": str(after_path),
                        "possible_elements": ["Bread roll"],
                        "category": "test_meal",
                    }
                ]
            ),
            encoding="utf-8",
        )
        (masks_dir / "before.json").write_text(
            json.dumps({"detections": [{"class": "bread_roll"}]}),
            encoding="utf-8",
        )
        (masks_dir / "after.json").write_text(
            json.dumps(
                {
                    "detections": [
                        {"class": "bread_roll"},
                        {"class": "fish_salmon"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        session = review_ui.ReviewSession(
            index_path,
            after_dir,
            output_path,
            before_dir,
            pairs_dir,
            masks_dir,
        )
        return session, output_path

    @staticmethod
    def record():
        return {
            "task_id": 1,
            "source_file": "def__after.jpg",
            "polygon_labels": ["bread_roll"],
            "numbers": {"pct_bread_roll": [50]},
            "choices": {
                "quality_flags": [["Obstacle obscuring food (cutlery, napkin, foil)"]]
            },
        }

    def test_quality_and_after_only_class_are_flagged(self):
        session, _ = self.build_session([self.record()])
        issues = session.validate_record("after", session.indexed["after"])
        checks = {issue["check"] for issue in issues}
        self.assertIn("quality_flag", checks)
        self.assertIn("unexpected_class_appears_after", checks)
        self.assertEqual(session.meta()["quality_count"], 1)
        self.assertIsNotNone(session.task_detail("after")["before_image_url"])

    def test_annotation_alias_does_not_create_false_zero_error(self):
        record = self.record()
        record["numbers"] = {"pct_vanilla_pudding": [0]}
        record["polygon_labels"] = ["vanilla_pudding_with_fruits"]
        session, _ = self.build_session([record])
        checks = {
            issue["check"]
            for issue in session.validate_record("after", session.indexed["after"])
        }
        self.assertNotIn("zero_consumption_and_missing_mask", checks)

    def test_missing_quality_field_is_editable(self):
        record = self.record()
        record["choices"] = {}
        session, _ = self.build_session([record])
        detail = session.task_detail("after")
        quality = next(c for c in detail["choices"] if c["from_name"] == "quality_flags")
        self.assertTrue(quality["multiple"])
        self.assertEqual(quality["values"], [])
        self.assertIn("Blurry", quality["options"])
        self.assertIn(
            "Non-edible residuals present (bones, core, crumbs only)",
            quality["options"],
        )

    def test_quality_edit_survives_save_and_reload(self):
        record = self.record()
        record["choices"]["quality_flags"] = [[review_ui.CLEAN_QUALITY_FLAG]]
        session, output_path = self.build_session([record])
        session.update_task(
            "after",
            {"choices": {"quality_flags": ["Blurry"]}},
        )
        session.save()
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        selected = saved["indexed_by_image_stem"]["after"][0]["choices"][
            "quality_flags"
        ]
        self.assertEqual(selected, [["Blurry"]])
        issues = session.validate_record("after", session.indexed["after"])
        self.assertTrue(
            any(
                issue["check"] == "quality_flag" and issue["message"] == "Blurry"
                for issue in issues
            )
        )

    def test_save_preserves_additional_duplicate_records(self):
        first = self.record()
        second = dict(first, task_id=2)
        session, output_path = self.build_session([first, second])
        session.save()
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(len(saved["indexed_by_image_stem"]["after"]), 2)


if __name__ == "__main__":
    unittest.main()
