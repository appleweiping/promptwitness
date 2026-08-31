"""Unambiguous JSON Pointer paths for reports."""

from __future__ import annotations


def json_pointer(*segments: str | int) -> str:
    """Encode path segments according to RFC 6901 JSON Pointer escaping."""

    encoded = (str(segment).replace("~", "~0").replace("/", "~1") for segment in segments)
    return "/" + "/".join(encoded)
