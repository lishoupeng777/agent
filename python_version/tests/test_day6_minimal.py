from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from app.batch import batch_evaluate
from app.models import EvalRequest
from app.reporter import run_full_evaluation
from main import app


client = TestClient(app)


def _make_request(request_id: str = "req-1", with_human_label: bool = True) -> EvalRequest:
    payload = {
        "request_id": request_id,
        "before_text": "原文第一段。\n原文第二段。",
        "after_text": "改写后第一段。\n改写后第二段。",
    }
    if with_human_label:
        payload["human_label"] = {
            "overall_score": 0.8,
            "flaws": [],
        }
    return EvalRequest(**payload)


class Day6MinimalTests(unittest.TestCase):
    def test_health_endpoint_returns_ok(self) -> None:
        response = client.get("/api/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "service": "llm-quality-judge"})

    def test_reproduce_endpoint_returns_not_found_payload_for_unknown_token(self) -> None:
        response = client.get("/api/v1/reproduce/not-exists-token")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["found"])
        self.assertEqual(body["token"], "not-exists-token")
        self.assertIn("未找到对应历史记录", body["message"])

    def test_history_endpoint_returns_basic_structure(self) -> None:
        response = client.get("/api/v1/history")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body.keys()), {"total_in_db", "returned", "offset", "records"})
        self.assertIsInstance(body["records"], list)
        self.assertIsInstance(body["returned"], int)
        self.assertIsInstance(body["offset"], int)

    def test_batch_evaluate_endpoint_rejects_more_than_50_requests(self) -> None:
        requests = [
            _make_request(request_id=f"req-{i}", with_human_label=False).model_dump(mode="json")
            for i in range(51)
        ]

        response = client.post("/api/v1/batch/evaluate", json=requests)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["error"], "单次批量评估最多支持50条请求")
        self.assertEqual(body["count"], 51)

    def test_run_full_evaluation_with_empty_requests_returns_complete_shell(self) -> None:
        report = run_full_evaluation([])

        self.assertEqual(report["report_meta"]["total_samples"], 0)
        self.assertEqual(report["per_sample_results"], [])
        self.assertEqual(report["calibration"], {})
        self.assertEqual(report["stability_summary"], {})
        self.assertEqual(report["flaw_metrics"], {})
        self.assertEqual(report["anchor_metrics"], {})
        self.assertEqual(report["bias_analysis"], {})
        self.assertTrue(report["reproducibility_verification"]["all_reproducible"])
        self.assertIsInstance(report["checklist"], list)

    def test_run_full_evaluation_without_ground_truth_flaws_does_not_force_metrics(self) -> None:
        requests = [
            _make_request("req-no-flaws-1", with_human_label=True),
            _make_request("req-no-flaws-2", with_human_label=True),
        ]

        report = run_full_evaluation(requests)

        self.assertEqual(report["report_meta"]["total_samples"], 2)
        self.assertEqual(len(report["per_sample_results"]), 2)
        self.assertIsInstance(report["calibration"], dict)
        self.assertEqual(report["flaw_metrics"], {})
        self.assertEqual(report["anchor_metrics"], {})

    def test_batch_evaluate_raises_value_error_when_request_count_exceeds_limit(self) -> None:
        requests = [_make_request(request_id=f"req-{i}", with_human_label=False) for i in range(51)]

        with self.assertRaises(ValueError) as ctx:
            batch_evaluate(requests)

        self.assertEqual(str(ctx.exception), "单次批量评估最多支持50条请求")


if __name__ == "__main__":
    unittest.main()
