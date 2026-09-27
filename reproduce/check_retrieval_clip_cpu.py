"""Real pretrained encoder on authored pattern images, not benchmark results."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from reproduce.retrieval_clip_cpu import ClipCPUEncoder, require_cpu
from reproduce.retrieval_cpu_ranking import rank_float32_cpu


def check(checkpoint: Path, output: Path) -> dict:
    require_cpu()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    encoder = None
    result = {
        "format": "promptwitness.clip-cpu-mechanical-qualification/v1",
        "status": "RUNNING",
        "benchmark_images_or_queries": False,
        "science_result": False,
        "captioner_or_LLM_called": False,
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
        # Authored deterministic input, not downloaded benchmark/demo images.
        for color, size, mode in (("red", (301, 225), "RGB"), ("blue", (225, 301), "RGBA")):
            image = Image.new(mode, size, "white")
            ImageDraw.Draw(image).rectangle((70, 70, 150, 150), fill=color)
            images.append(image)
        grey = Image.new("L", (240, 240), 128)
        images.append(grey)
        texts = ("a red square on a white background", "a blue square on a white background")
        encoder = ClipCPUEncoder(checkpoint)
        gallery = torch.stack([encoder.encode_image(x) for x in images])
        queries = [encoder.encode_text(x) for x in texts]
        # An unrelated intervening item and reversed order exercise batch1 reuse.
        repeat_images = [encoder.encode_image(x) for x in reversed(images)]
        repeat_queries = [encoder.encode_text(x) for x in reversed(texts)]
        for original, repeat in zip(gallery, reversed(repeat_images), strict=True):
            if not torch.equal(original, repeat):
                raise ValueError("image features changed with single-item traversal order")
        for original, repeat in zip(queries, reversed(repeat_queries), strict=True):
            if not torch.equal(original, repeat):
                raise ValueError("text features changed with single-item traversal order")
        ids = ("authored-red", "authored-blue", "authored-grey")
        ranks = [rank_float32_cpu(q, gallery, ids) for q in queries]
        reranks = [rank_float32_cpu(q, gallery, ids) for q in reversed(repeat_queries)]
        if ranks != reranks:
            raise ValueError("full ranks changed after identical singleton encoding")
        rejected = []
        for label, text in (("overlength", "square " * 100), ("empty", "")):
            before = encoder.costs["text_forward_calls"]
            try:
                encoder.encode_text(text)
            except (RuntimeError, ValueError) as error:
                if label == "overlength" and "too long" not in str(error):
                    raise
                if encoder.costs["text_forward_calls"] != before:
                    raise ValueError("invalid text reached the model") from error
                rejected.append(label)
            else:
                raise ValueError("invalid text accepted without an error")
        if rejected != ["overlength", "empty"]:
            raise ValueError("expected both input rejection paths")
        result.update(
            status="PASS_AUTHORED_IMAGES_REAL_PRETRAINED_ENCODER_CPU_ONLY",
            checkpoint_sha256=encoder.checkpoint_sha256,
            input_resolution=encoder.model.visual.input_resolution,
            context_length=encoder.model.context_length,
            dtype=str(gallery.dtype),
            device=str(gallery.device),
            gallery_shape=list(gallery.shape),
            full_ranks=ranks,
            image_repeats_bitwise_equal=3,
            text_repeats_bitwise_equal=2,
            full_ranks_equal=2,
            input_rejections=rejected,
            model_load_wall_seconds=encoder.load_wall_seconds,
            torch=torch.__version__,
            torch_threads=torch.get_num_threads(),
            trainable_parameters=sum(
                p.numel() for p in encoder.model.parameters() if p.requires_grad
            ),
            inference_costs=encoder.costs,
            fusion_or_caption_policy_frozen=False,
            semantic_accuracy_or_benchmark_parity_established=False,
            tolerance_or_hyperparameter_tuning=False,
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
    result = check(args.checkpoint, args.new_output)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
