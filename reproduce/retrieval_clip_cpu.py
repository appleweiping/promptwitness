"""Offline, single-item CLIP ViT-L/14 encoding for research qualification.

Uses OpenAI's separately staged MIT implementation and trusted official weights.
No downloader, model substitution, truncation or implicit GPU/batch selection.
This is not a captioner or a validated CIR system.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

CHECKPOINT_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"


def require_cpu() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("empty CUDA visibility required for the CPU encoder")


def verify_checkpoint(checkpoint: Path) -> str:
    """Verify the identity already specified by the first-party loader map."""
    require_cpu()
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != CHECKPOINT_SHA256:
        raise ValueError("checkpoint differs from the official ViT-L/14 artifact")
    return actual


class ClipCPUEncoder:
    """A fixed float32/eval/no-grad batch1 path; failures propagate to caller.

    Every item is encoded alone, including gallery construction. This trades
    throughput for a fixed dispatch shape in full vs. adaptive completion.
    Image features stay raw for the ranker's one gallery normalization. Text
    queries are normalized exactly once here; the ranker does not renormalize them.
    Calls count attempted forwards; preprocessing/tokenization failures count
    input attempts but not forwards. Timing covers these stages, not all costs.
    """

    def __init__(self, checkpoint: Path):
        require_cpu()
        self.costs = {
            "image_attempts": 0,
            "text_attempts": 0,
            "image_forward_calls": 0,
            "text_forward_calls": 0,
            "image_completed": 0,
            "text_completed": 0,
            "image_wall_seconds": 0.0,
            "text_wall_seconds": 0.0,
        }
        started = time.monotonic()
        self.checkpoint_sha256 = verify_checkpoint(checkpoint)
        import clip
        import torch

        self.clip, self.torch = clip, torch
        self.model, self.preprocess = clip.load(str(checkpoint), device="cpu", jit=False)
        self.model.eval()
        self.model.requires_grad_(False)
        if self.model.visual.input_resolution != 224 or self.model.context_length != 77:
            raise ValueError("expected original ViT-L/14 224px/77-token architecture")
        self.load_wall_seconds = time.monotonic() - started

    def _validate(self, features):
        torch = self.torch
        if (
            features.shape != (1, 768)
            or features.device.type != "cpu"
            or features.dtype != torch.float32
            or features.requires_grad
            or not torch.isfinite(features).all()
        ):
            raise ValueError("finite no-grad float32 CPU 1x768 encoder output required")
        norm = torch.linalg.vector_norm(features, dim=-1, keepdim=True)
        if not torch.isfinite(norm).all() or not (norm >= 1e-12).all():
            raise ValueError("nondegenerate encoder output required")
        return norm

    def _unit(self, features):
        return (features / self._validate(features))[0]

    def encode_image(self, image):
        require_cpu()
        self.costs["image_attempts"] += 1
        started = time.monotonic()
        try:
            tensor = self.preprocess(image).unsqueeze(0)
            with self.torch.no_grad():
                self.costs["image_forward_calls"] += 1
                features = self.model.encode_image(tensor)
                self._validate(features)
                result = features[0]
            self.costs["image_completed"] += 1
            return result
        finally:
            self.costs["image_wall_seconds"] += time.monotonic() - started

    def encode_text(self, text: str):
        require_cpu()
        self.costs["text_attempts"] += 1
        started = time.monotonic()
        try:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("nonempty text string required")
            tokens = self.clip.tokenize([text], context_length=77, truncate=False)
            with self.torch.no_grad():
                self.costs["text_forward_calls"] += 1
                result = self._unit(self.model.encode_text(tokens))
            self.costs["text_completed"] += 1
            return result
        finally:
            self.costs["text_wall_seconds"] += time.monotonic() - started
