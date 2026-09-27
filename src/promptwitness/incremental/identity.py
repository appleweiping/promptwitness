"""Complete execution identity; independently generated replicates never alias."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .sampling import digest


@dataclass(frozen=True, slots=True)
class ExecutionIdentity:
    model_revision: str
    tokenizer_revision: str
    backend_version: str
    backend_config_digest: str
    template_digest: str
    decoding_digest: str
    scorer_digest: str
    data_digest: str
    tool_environment_digest: str

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value for value in asdict(self).values()):
            raise ValueError("all execution identity fields are required")
        for name, value in asdict(self).items():
            if name.endswith("digest") and (
                len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("SHA-256 execution digest required")

    @property
    def sha256(self) -> str:
        return digest(asdict(self))

    def request_key(self, request: Any, *, unit_id: str, replicate_id: str) -> str:
        if not unit_id or not replicate_id:
            raise ValueError("unit and recorded random replicate identity required")
        return digest(
            {
                "execution": asdict(self),
                "complete_request": request,
                "unit_id": unit_id,
                "replicate_id": replicate_id,
            }
        )
