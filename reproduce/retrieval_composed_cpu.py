"""Reference-image plus modification-text direct CLIP fusion baseline.

Mechanical research baseline only. The caller must freeze a fusion weight on
training-side data before any benchmark final evaluation. This module ranks the
full gallery; CIRR reference exclusion belongs to its scorer, not this ranker.
"""

from __future__ import annotations

import math

from reproduce.retrieval_clip_cpu import require_cpu
from reproduce.retrieval_cpu_ranking import rank_float32_cpu


def direct_fusion_query(reference, modification, image_weight: float):
    """Return one unit float32 CPU query; text is already unit from ClipCPUEncoder."""
    require_cpu()
    if (
        not isinstance(image_weight, float)
        or not math.isfinite(image_weight)
        or not 0.0 <= image_weight <= 1.0
    ):
        raise ValueError("finite frozen image weight in [0, 1] required")

    import torch
    import torch.nn.functional as F

    if (
        not isinstance(reference, torch.Tensor)
        or not isinstance(modification, torch.Tensor)
        or reference.shape != (768,)
        or modification.shape != (768,)
        or reference.device.type != "cpu"
        or modification.device.type != "cpu"
        or reference.dtype != torch.float32
        or modification.dtype != torch.float32
        or reference.requires_grad
        or modification.requires_grad
        or not torch.isfinite(reference).all()
        or not torch.isfinite(modification).all()
    ):
        raise ValueError("finite no-grad float32 CPU 768-vectors required")
    reference_norm = torch.linalg.vector_norm(reference)
    modification_norm = torch.linalg.vector_norm(modification)
    if (
        not torch.isfinite(reference_norm)
        or not torch.isfinite(modification_norm)
        or reference_norm < 1e-12
        or not torch.isclose(
            modification_norm,
            torch.tensor(1.0, dtype=torch.float32, device="cpu"),
            rtol=0,
            atol=1e-4,
        )
    ):
        raise ValueError("nonzero reference and already-unit text modification required")
    with torch.no_grad():
        image_unit = F.normalize(reference, dim=0)
        mixed = image_weight * image_unit + (1.0 - image_weight) * modification
        mixed_norm = torch.linalg.vector_norm(mixed)
        if not torch.isfinite(mixed_norm) or mixed_norm < 1e-12:
            raise ValueError("degenerate direct-fusion query")
        return mixed / mixed_norm


def rank_direct_composed(
    gallery,
    candidate_ids: tuple[str, ...],
    reference_id: str,
    modification,
    image_weight: float,
) -> tuple[str, ...]:
    """Rank complete gallery; do not silently exclude the reference item."""
    require_cpu()
    if reference_id not in candidate_ids:
        raise ValueError("reference ID must be present in the complete gallery")
    query = direct_fusion_query(
        gallery[candidate_ids.index(reference_id)], modification, image_weight
    )
    return rank_float32_cpu(query, gallery, candidate_ids)
