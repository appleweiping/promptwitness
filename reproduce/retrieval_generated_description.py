"""Input-only caption + modification -> metered model -> restricted CLIP ranker.

This is a search-query transport, not a qualified captioner or an online
experiment. The caller must establish caption provenance, frozen execution,
data access, ranker identity, and the continued physical resource ledger.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path

from promptwitness.parser import parse_prompt
from promptwitness.variables import inspect_variables, render_template
from reproduce.retrieval_direct_clip_ranker import load_input_only
from reproduce.torch_runtime import RETRIEVAL_DESCRIPTION_CAP


def description_wire(
    candidate: dict,
    *,
    query_id: str,
    reference_caption: str,
    modification: str,
    replicate: str,
) -> dict:
    """Render only the two input-side fields, with no image or gold access.

    A candidate may bind both fields explicitly or use a fixed appended user
    message. The latter preserves existing instruction/demo-prefix prompts.
    """
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (query_id, reference_caption, modification, replicate)
    ):
        raise ValueError("nonempty literal query, caption, modification and replicate required")
    document = parse_prompt(candidate)
    if (
        document.tools
        or not document.messages
        or any(
            message.content_parts
            or message.name is not None
            or message.role not in {"system", "user", "assistant"}
            for message in document.messages
        )
    ):
        raise ValueError("description prompt supports text messages only")
    inventories = [inspect_variables(message.content) for message in document.messages]
    if any(inventory.malformed for inventory in inventories):
        raise ValueError("malformed description prompt variable")
    variables = set().union(*(inventory.names for inventory in inventories))
    if variables not in (set(), {"reference_caption", "modification"}):
        raise ValueError("description prompt must bind both input fields or neither")
    values = {"reference_caption": reference_caption, "modification": modification}
    messages = [
        {"role": message.role, "content": render_template(message.content, values)}
        for message in document.messages
    ]
    if not variables:
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Reference image caption: {reference_caption}\n"
                    f"Requested modification: {modification}\n"
                    "Describe the target image."
                ),
            }
        )
    if messages[-1]["role"] != "user":
        raise ValueError("description prompt must end in a user message")
    return {
        "id": query_id,
        "replicate": replicate,
        "role": "task",
        "family": "cir_description",
        "messages": messages,
        "max_new_tokens": RETRIEVAL_DESCRIPTION_CAP,
    }


class GeneratedDescriptionRanker:
    """Expose ``rank_one`` for RetrievalAuditBridge using actual model output.

    The model's ``metered`` call reserves before inference and retains its
    response/cost on failure. The ranker's separate request ID then charges
    CLIP text encoding and complete-gallery ranking. Reusing this instance's
    attempt prefix is a replay error, never a free cache hit.
    """

    def __init__(
        self,
        *,
        input_dir: Path,
        dataset: str,
        captions: Mapping[str, str],
        candidate: dict,
        model,
        ranker,
        execution: dict,
        replicate: str,
        attempt_prefix: str,
    ) -> None:
        if not isinstance(attempt_prefix, str) or not attempt_prefix.strip():
            raise ValueError("nonempty attempt prefix required")
        queries, _ = load_input_only(input_dir, dataset)
        references = {reference_id for reference_id, _, _ in queries.values()}
        if set(captions) != references or any(
            not isinstance(value, str) or not value.strip() for value in captions.values()
        ):
            raise ValueError("one nonempty fixed caption per reference image required")
        self.queries = queries
        self.captions = dict(captions)
        self.candidate = copy.deepcopy(candidate)
        self.model = model
        self.ranker = ranker
        self.execution = copy.deepcopy(execution)
        self.replicate = replicate
        self.attempt_prefix = attempt_prefix

    def rank_one(self, query_id: str) -> tuple[str, ...]:
        if query_id not in self.queries:
            raise ValueError("query is outside the input-only population")
        reference_id, modification, _ = self.queries[query_id]
        wire = description_wire(
            self.candidate,
            query_id=query_id,
            reference_caption=self.captions[reference_id],
            modification=modification,
            replicate=self.replicate,
        )
        response = self.model.metered(
            f"{self.attempt_prefix}:generate:{query_id}",
            self.execution,
            wire,
            purpose="search",
        )
        if (
            not isinstance(response, dict)
            or response.get("id") != query_id
            or response.get("replicate") != self.replicate
            or response.get("status") != "completed"
            or not isinstance(response.get("output"), str)
            or not response["output"].strip()
        ):
            raise ValueError("metered description response is incomplete or empty")
        return self.ranker.rank_description(
            query_id, f"{self.attempt_prefix}:rank:{query_id}", response["output"]
        )
