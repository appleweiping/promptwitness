"""Bounded, read-only checks before downloading any model weights."""

from __future__ import annotations

import importlib
import json
import time
import urllib.error
import urllib.request


def main() -> None:
    result: dict[str, object] = {"format": "promptwitness.delta.inference-preflight/v1"}
    for module_name, names in (
        ("torch", ("cuda",)),
        ("transformers", ("Qwen3_5ForConditionalGeneration", "Olmo3ForCausalLM")),
    ):
        try:
            module = importlib.import_module(module_name)
            for name in names:
                value = getattr(module, name)
                result[f"{module_name}.{name}"] = (
                    {"available": value.is_available(), "device_count": value.device_count()}
                    if name == "cuda"
                    else {"imported": True}
                )
        except Exception as error:
            result[module_name] = {"error_type": type(error).__name__, "message": str(error)}
    for endpoint in ("https://huggingface.co", "https://hf-mirror.com"):
        started = time.monotonic()
        request = urllib.request.Request(endpoint, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                result[endpoint] = {"status": response.status}
        except (OSError, urllib.error.HTTPError) as error:
            result[endpoint] = {"error_type": type(error).__name__}
        result[f"{endpoint}.elapsed_seconds"] = time.monotonic() - started
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
