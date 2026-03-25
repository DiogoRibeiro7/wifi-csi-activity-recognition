"""Tests for utility I/O helper functions."""

import json

from wifi_activity_recognition.utils.io import save_evaluation_results


def test_save_evaluation_results_writes_json(tmp_path):
    """Evaluation results should be persisted as JSON."""
    output_path = tmp_path / "evaluation.json"
    results = {
        "accuracy": 0.91,
        "precision": 0.88,
        "per_class_metrics": {"walking": {"f1_score": 0.9}},
    }

    save_evaluation_results(results, output_path)

    with open(output_path, "r", encoding="utf-8") as handle:
        saved = json.load(handle)

    assert saved == results
