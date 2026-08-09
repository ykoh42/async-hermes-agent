"""Stdio worker for the pinned Mem0 transformations."""

from __future__ import annotations

import json
import sys
from typing import Any


def _execute(request: dict[str, Any]) -> Any:
    operation = request.get("operation")
    if operation == "lemmatize":
        from mem0.utils.lemmatization import lemmatize_for_bm25

        return lemmatize_for_bm25(request["text"])
    if operation == "extract":
        from mem0.utils.entity_extraction import extract_entities

        return extract_entities(request["text"])
    if operation == "extract_batch":
        from mem0.utils.entity_extraction import extract_entities_batch

        return extract_entities_batch(request["texts"])
    raise ValueError(f"Unsupported Mem0 transform operation: {operation}")


def _serve_stdio() -> None:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("operation") == "close":
                return
            response = {"result": _execute(request)}
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__" and sys.argv[1:] == ["--stdio"]:
    _serve_stdio()
