from __future__ import annotations

from pathlib import Path

import requests

from src.utils.io import write_json


class WorkflowAgentClient:
    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def create_workflow_from_doc(self, doc_result: dict) -> dict:
        response = requests.post(f"{self.base_url}/workflows", json=doc_result, timeout=self.timeout)
        response.raise_for_status()
        return response.json()


def save_offline_payload(payload: dict, path: str | Path = "outputs/workflow_payloads/payload.json") -> Path:
    return write_json(path, payload)
