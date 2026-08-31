"""Template-variable extraction and rendering checks."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

_VARIABLE = re.compile(r"(?<!\{)\{\{\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*\}\}(?!\})")
_OPEN = re.compile(r"\{\{")
_CLOSE = re.compile(r"\}\}")


@dataclass(frozen=True, slots=True)
class VariableInventory:
    """Variables and occurrence counts extracted from one document."""

    counts: dict[str, int]
    malformed: bool = False

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self.counts)


def inspect_variables(text: str) -> VariableInventory:
    """Extract ``{{ variable }}`` names without evaluating the template."""

    names = _VARIABLE.findall(text)
    remainder = _VARIABLE.sub("", text)
    malformed = bool(_OPEN.search(remainder) or _CLOSE.search(remainder))
    return VariableInventory(dict(Counter(names)), malformed=malformed)


def render_template(text: str, values: Mapping[str, object], *, strict: bool = True) -> str:
    """Render supported variables with plain string substitution.

    This intentionally is not a general template engine: values cannot execute
    expressions, filters, attribute lookups, or code.
    """

    inventory = inspect_variables(text)
    if inventory.malformed:
        raise ValueError("template contains malformed or unsupported variable syntax")
    missing = inventory.names - values.keys()
    if strict and missing:
        raise KeyError(f"missing template variables: {', '.join(sorted(missing))}")

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            return match.group(0)
        return str(values[name])

    return _VARIABLE.sub(replace, text)
