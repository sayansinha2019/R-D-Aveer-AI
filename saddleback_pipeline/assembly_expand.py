"""Assembly-aware BOM expansion (shop: main member + clips + bolts + plates).

**Not auto-run:** fabricators derive connection kits from detailing software or
engineer-approved libraries. This module is a **hook** for future rule packs
(YAML/JSON keyed by connection type) so the pipeline can emit ``Main`` / ``Category``
rows aligned with Tekla-style exports *after* you supply authoritative rules.

Without validated rules, expanding assemblies would invent hardware — that is why
the default integrated pipeline does **not** call these functions.
"""

from __future__ import annotations

from typing import Any


def expand_assembly_placeholders(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """No-op pass-through today. Replace with rule-driven expansion when config exists."""
    return list(entities)


def load_connection_rules_yaml(_path: str) -> dict[str, Any]:
    """Reserved: load bolt/clip patterns per connection family."""
    return {}
