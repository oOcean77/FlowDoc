from __future__ import annotations


def build_workflow_payload(extracted_fields: dict, doc_type: str, user_id: str) -> dict:
    if doc_type == "access_request_form":
        applicant = extracted_fields.get("applicant") or user_id
        scope = extracted_fields.get("permission_scope", "unknown_scope")
        reason = extracted_fields.get("reason", "document request")
        deadline = extracted_fields.get("deadline", "unspecified")
        return {
            "user_id": applicant,
            "message": f"Please request {scope} permission, reason: {reason}, deadline: {deadline}",
            "source": "FlowDoc-VLM",
            "doc_type": doc_type,
        }
    return {
        "user_id": user_id,
        "message": f"Please review extracted {doc_type} fields: {extracted_fields}",
        "source": "FlowDoc-VLM",
        "doc_type": doc_type,
    }
