"""Synthetic completed envelopes for tests only, never real model-output evidence.

Production code must not import this helper or manufacture a missing completion
status. These fixtures exercise deterministic contract/state/storage behavior.
"""

from __future__ import annotations

import json
from typing import Any

from promptwitness.interview_models import BoundEvidence
from promptwitness.interview_stages import AnalysisResult, QuestionResult


def completion_envelope(result: AnalysisResult | QuestionResult) -> dict[str, Any]:
    """Encode an already chosen test result as a scripted positive completion."""
    if isinstance(result, QuestionResult):
        body = {
            "format": "promptwitness.question-response/v1",
            "binding_digest": result.request.binding_digest,
            "target": result.target.link.to_dict(),
            "text": result.text,
            "memory_ids": list(result.memory_ids),
        }
    elif isinstance(result, AnalysisResult):
        sources: dict[str, BoundEvidence] = {}
        for item in (*result.assessments, *result.memories, *result.proposals):
            for evidence in item.evidence:
                sources[evidence.evidence_id] = evidence
        names = {evidence_id: f"e{index}" for index, evidence_id in enumerate(sorted(sources), 1)}

        def references(values: tuple[BoundEvidence, ...]) -> list[str]:
            return [names[item.evidence_id] for item in values]

        proposal = result.proposal
        body = {
            "format": "promptwitness.analysis-response/v1",
            "binding_digest": result.request.binding_digest,
            "evidence": [
                {
                    "id": names[key],
                    "answer_id": value.answer.answer_id,
                    "start": value.reference.start,
                    "end": value.reference.end,
                    "quote": value.reference.quote,
                }
                for key, value in sorted(sources.items())
            ],
            "assessments": [
                {
                    "topic_id": item.link.topic_id,
                    "criterion_id": item.link.criterion_id,
                    "status": item.status,
                    "evidence_ids": references(item.evidence),
                    "rationale": item.rationale,
                }
                for item in result.assessments
            ],
            "memories": [
                {
                    "links": [link.to_dict() for link in item.links],
                    "evidence_ids": references(item.evidence),
                    "summary": {"kind": "model_proposed", "text": item.model_summary},
                }
                for item in result.memories
            ],
            "proposal": {
                "topic": proposal.topic.to_dict(),
                "parent_topic_id": proposal.parent_topic_id,
                "evidence_ids": references(proposal.evidence),
            }
            if proposal
            else None,
        }
    else:
        raise TypeError("test fixture requires a validated analysis/question result")
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(body, ensure_ascii=False, allow_nan=False),
                },
            }
        ]
    }
