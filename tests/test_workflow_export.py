from __future__ import annotations

import json
from pathlib import Path

from src.integration.export_to_workflow import build_workflow_payload
from src.integration.workflow_client import save_offline_payload


def test_build_workflow_payload() -> None:
    payload = build_workflow_payload(
        {
            "applicant": "u_001",
            "permission_scope": "server_root",
            "reason": "troubleshoot payment failure",
            "deadline": "2026-06-05 22:00",
        },
        "access_request_form",
        "u_001",
    )
    assert payload["user_id"] == "u_001"
    assert "server_root" in payload["message"]


def test_workflow_client_offline_fallback() -> None:
    root = Path("outputs/test_artifacts")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "payload.json"
    save_offline_payload({"user_id": "u_001", "message": "hello"}, path)
    assert json.loads(path.read_text(encoding="utf-8"))["user_id"] == "u_001"
