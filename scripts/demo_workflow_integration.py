from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.integration.export_to_workflow import build_workflow_payload
from src.integration.workflow_client import WorkflowAgentClient, save_offline_payload


if __name__ == "__main__":
    extracted = {
        "applicant": "u_001",
        "permission_scope": "server_root",
        "reason": "troubleshoot payment failure",
        "deadline": "2026-06-05 22:00",
    }
    payload = build_workflow_payload(extracted, "access_request_form", "u_001")
    base_url = os.getenv("WORKFLOW_AGENT_URL", "").strip()
    if base_url:
        result = WorkflowAgentClient(base_url).create_workflow_from_doc(payload)
        print(result)
    else:
        path = save_offline_payload(payload)
        print(f"Saved offline workflow payload to {path}")
        print(payload)
