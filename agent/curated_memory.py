"""Read-only lexical search over the conservative Railway gbrain snapshot."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def search_curated_memory(bundle_path: Path, query: str, *, limit: int = 5) -> str:
    """Return compact ranked hits, or a fail-closed availability message."""
    try:
        bundle: dict[str, Any] = json.loads(bundle_path.read_text(encoding="utf-8"))
        documents = bundle["documents"]
        if bundle.get("schema_version") != 1 or not isinstance(documents, list):
            raise ValueError("invalid curated bundle")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return "(memoria curada no disponible)"

    terms = [term.lower() for term in re.findall(r"[\wáéíóúñ]{2,}", query, re.IGNORECASE)]
    if not terms:
        return "(consulta de memoria vacía)"
    ranked: list[tuple[int, dict[str, Any]]] = []
    for document in documents:
        haystack = " ".join(str(document.get(key, "")) for key in ("slug", "title", "content", "timeline")).lower()
        score = sum(haystack.count(term) for term in terms)
        if score:
            ranked.append((score, document))
    if not ranked:
        return "(sin resultados en la memoria curada)"
    ranked.sort(key=lambda item: (-item[0], str(item[1].get("slug", ""))))
    hits: list[str] = []
    for _score, document in ranked[:limit]:
        body = (str(document.get("content", "")) + "\n" + str(document.get("timeline", ""))).strip()
        body = re.sub(r"\s+", " ", body)[:420]
        hits.append(f"[{document.get('slug', 'sin-slug')}] {document.get('title', '')}\n{body}")
    return "\n\n".join(hits)[:1800]
