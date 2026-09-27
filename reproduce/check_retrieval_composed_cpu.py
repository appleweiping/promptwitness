"""Run an actual reference-image plus text-to-gallery path on authored images."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from reproduce.retrieval_clip_cpu import ClipCPUEncoder, require_cpu
from reproduce.retrieval_composed_cpu import rank_direct_composed


def check(checkpoint: Path, output: Path) -> dict:
    require_cpu()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    encoder = None
    result = {
        "format": "promptwitness.composed-cpu-mechanical-qualification/v1",
        "status": "RUNNING",
        "benchmark_images_or_queries": False,
        "science_result": False,
        "query_modalities": ["reference_image", "modification_text"],
        "captioner_or_LLM_called": False,
        "direct_fusion_image_weight": 0.5,
        "weight_scientifically_frozen": False,
        "gpu_allocation": False,
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "query_and_image_batch": 1,
    }
    try:
        import torch
        from PIL import Image, ImageDraw

        torch.set_num_threads(1)
        torch.manual_seed(11)
        images = []
        for color, size, mode in (("red", (301, 225), "RGB"), ("blue", (225, 301), "RGBA")):
            image = Image.new(mode, size, "white")
            ImageDraw.Draw(image).rectangle((70, 70, 150, 150), fill=color)
            images.append(image)
        images.append(Image.new("L", (240, 240), 128))
        ids = ("authored-red", "authored-blue", "authored-grey")
        cases = (
            ("authored-red", "make the square blue"),
            ("authored-blue", "make the square red"),
        )
        encoder = ClipCPUEncoder(checkpoint)
        gallery = torch.stack([encoder.encode_image(image) for image in images])
        modifications = [encoder.encode_text(text) for _, text in cases]
        ranks = [
            rank_direct_composed(gallery, ids, ref, text, 0.5)
            for (ref, _), text in zip(cases, modifications, strict=True)
        ]
        reversed_gallery = [encoder.encode_image(image) for image in reversed(images)]
        reversed_modifications = [encoder.encode_text(text) for _, text in reversed(cases)]
        for original, repeated in zip(gallery, reversed(reversed_gallery), strict=True):
            if not torch.equal(original, repeated):
                raise ValueError("image encoding changed under reversed singleton traversal")
        for original, repeated in zip(modifications, reversed(reversed_modifications), strict=True):
            if not torch.equal(original, repeated):
                raise ValueError("modification encoding changed under reversed traversal")
        reranks = [
            rank_direct_composed(gallery, ids, ref, text, 0.5)
            for (ref, _), text in zip(cases, reversed(reversed_modifications), strict=True)
        ]
        if ranks != reranks:
            raise ValueError("complete composed rankings changed under repeat")
        result.update(
            status="PASS_AUTHORED_COMPOSED_IMAGE_TEXT_CPU_ONLY",
            checkpoint_sha256=encoder.checkpoint_sha256,
            gallery_shape=list(gallery.shape),
            dtype=str(gallery.dtype),
            device=str(gallery.device),
            full_ranks=ranks,
            image_repeats_bitwise_equal=3,
            modification_repeats_bitwise_equal=2,
            full_ranks_equal=2,
            image_forward_calls=encoder.costs["image_forward_calls"],
            text_forward_calls=encoder.costs["text_forward_calls"],
            inference_costs=encoder.costs,
            model_load_wall_seconds=encoder.load_wall_seconds,
            reference_exclusion="NOT_APPLIED_HERE_SCORER_SPECIFIC",
            fusion_hyperparameter_tuning=False,
            semantic_accuracy_or_benchmark_parity_established=False,
        )
        return result
    except BaseException as error:
        result.update(status="FAILED_RETAINED", error_type=type(error).__name__, error=str(error))
        if encoder is not None:
            result["inference_costs"] = encoder.costs
        raise
    finally:
        result["checker_wall_seconds"] = time.monotonic() - started
        (output / "qualification.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("new_output", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.checkpoint, args.new_output), allow_nan=False))


if __name__ == "__main__":
    main()
