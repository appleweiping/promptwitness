"""Single-query float32 CPU distance ranking for the ICMR research backend.

Preserve gallery row order and Torch's distance argsort, not lexical ties.
This reproduces the selected published kernels' arithmetic for float32 inputs,
not their whole CLI, GPU/batched execution, image encoder or benchmark results.
The prediction producer normalizes the query once; do not normalize it again.
"""

from __future__ import annotations

import os


def rank_float32_cpu(query, gallery, candidate_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Rank one already-unit query against the complete gallery on CPU only.

    Single-query execution is fixed, including full/survivor scoring, so changing
    the adaptive query batch cannot change this matmul shape. Embedding generation
    still needs its own online qualification. Invalid features fail, never score0.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("empty CUDA visibility required for the CPU ranking backend")
    if (
        not isinstance(candidate_ids, tuple)
        or not candidate_ids
        or any(not isinstance(x, str) or not x for x in candidate_ids)
        or len(set(candidate_ids)) != len(candidate_ids)
    ):
        raise ValueError("ordered unique immutable candidate IDs required")
    import torch
    import torch.nn.functional as F

    if any(
        not isinstance(x, torch.Tensor) or x.device.type != "cpu" or x.dtype != torch.float32
        for x in (query, gallery)
    ):
        raise ValueError("float32 CPU tensors required; no implicit device/dtype conversion")
    if (
        query.ndim != 1
        or gallery.ndim != 2
        or query.numel() == 0
        or gallery.shape != (len(candidate_ids), query.numel())
    ):
        raise ValueError("one query and the exact complete gallery shape required")
    if not torch.isfinite(query).all() or not torch.isfinite(gallery).all():
        raise ValueError("finite embedding features required")
    query_norm = torch.linalg.vector_norm(query)
    gallery_norms = torch.linalg.vector_norm(gallery, dim=-1)
    if (
        not torch.isclose(
            query_norm, torch.tensor(1.0, dtype=torch.float32, device="cpu"), rtol=0, atol=1e-4
        )
        or not torch.isfinite(gallery_norms).all()
        or not (gallery_norms >= 1e-12).all()
    ):
        raise ValueError("unit query and nondegenerate gallery vectors required")
    with torch.no_grad():
        normalized_gallery = F.normalize(gallery, dim=-1)
        # Sorting 1-dot is intentional: float32 subtraction can merge distinct
        # dot products. Sorting -dot or lexical IDs is not the same operation.
        distances = 1 - query.unsqueeze(0) @ normalized_gallery.T
        indices = torch.argsort(distances, dim=-1)[0].tolist()
    return tuple(candidate_ids[i] for i in indices)
