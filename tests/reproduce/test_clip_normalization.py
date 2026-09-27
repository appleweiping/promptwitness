"""Regression for routing raw gallery features vs. normalized text queries."""

from contextlib import nullcontext
from types import SimpleNamespace

from reproduce.retrieval_clip_cpu import ClipCPUEncoder


def test_only_query_uses_unit_normalizer(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    encoder = object.__new__(ClipCPUEncoder)
    raw_image = object()
    raw_text = object()
    unit_text = object()
    norm_calls = []
    validation_calls = []
    encoder.costs = {
        "image_attempts": 0,
        "text_attempts": 0,
        "image_forward_calls": 0,
        "text_forward_calls": 0,
        "image_completed": 0,
        "text_completed": 0,
        "image_wall_seconds": 0.0,
        "text_wall_seconds": 0.0,
    }
    encoder.torch = SimpleNamespace(no_grad=nullcontext)
    encoder.model = SimpleNamespace(
        encode_image=lambda _: (raw_image,), encode_text=lambda _: raw_text
    )
    encoder.preprocess = lambda _: SimpleNamespace(unsqueeze=lambda _: "singleton_image")
    encoder.clip = SimpleNamespace(tokenize=lambda *args, **kwargs: "singleton_text")
    encoder._validate = lambda features: validation_calls.append(features)

    def normalize(features):
        norm_calls.append(features)
        return unit_text

    encoder._unit = normalize
    assert encoder.encode_image("authored") is raw_image
    assert encoder.encode_text("authored") is unit_text
    assert validation_calls == [(raw_image,)]
    assert norm_calls == [raw_text]
    assert encoder.costs["image_completed"] == encoder.costs["text_completed"] == 1
