"""Parallel.ai web search + content extraction — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider` and uses the
native ``AsyncParallel`` SDK client for both search and extraction.

Config keys this provider responds to::

    web:
      search_backend: "parallel"      # explicit per-capability
      extract_backend: "parallel"     # explicit per-capability
      backend: "parallel"             # shared fallback
      # Optional: search mode (default "agentic"; also "fast" or "one-shot")
      # via the PARALLEL_SEARCH_MODE env var.

Env vars::

    PARALLEL_API_KEY=...             # https://parallel.ai (required)
    PARALLEL_SEARCH_MODE=agentic     # optional: agentic|fast|one-shot
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

async def _get_parallel_client() -> Any:
    """Lazy-load and cache the native async Parallel client."""
    import tools.web_tools as _wt

    cached = getattr(_wt, "_parallel_client", None)
    if cached is not None:
        return cached

    from agent.web_search_provider import get_provider_env

    api_key = await get_provider_env("PARALLEL_API_KEY")
    if not api_key:
        raise ValueError(
            "PARALLEL_API_KEY environment variable not set. "
            "Get your API key at https://parallel.ai"
        )

    try:
        from parallel import AsyncParallel  # noqa: WPS433 — deliberately lazy
    except ImportError as exc:
        raise ImportError(
            "The optional Parallel SDK is not installed. "
            "Install async-hermes-agent[parallel-web]."
        ) from exc

    client = AsyncParallel(api_key=api_key)
    _wt._parallel_client = client
    return client


def _reset_clients_for_tests() -> None:
    """Drop the cached client so tests can re-instantiate cleanly."""
    import tools.web_tools as _wt

    _wt._parallel_client = None


def _resolve_search_mode() -> str:
    """Return the validated PARALLEL_SEARCH_MODE value (default "agentic")."""
    mode = os.getenv("PARALLEL_SEARCH_MODE", "agentic").lower().strip()
    if mode not in {"fast", "one-shot", "agentic"}:
        mode = "agentic"
    return mode


class ParallelWebSearchProvider(WebSearchProvider):
    """Parallel.ai search + async extract provider."""

    @property
    def name(self) -> str:
        return "parallel"

    @property
    def display_name(self) -> str:
        return "Parallel"

    async def is_available(self) -> bool:
        """Return True when ``PARALLEL_API_KEY`` is set to a non-empty value."""
        from agent.web_search_provider import get_provider_env

        return bool(await get_provider_env("PARALLEL_API_KEY"))

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    async def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a Parallel search through the native async SDK.

        Uses the ``beta.search`` endpoint with the configured mode
        (``PARALLEL_SEARCH_MODE`` env var, default "agentic"). Limit is
        capped at 20 server-side.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            mode = _resolve_search_mode()
            logger.info(
                "Parallel search: '%s' (mode=%s, limit=%d)", query, mode, limit
            )
            client = await _get_parallel_client()
            response = await client.beta.search(
                search_queries=[query],
                objective=query,
                mode=mode,
                max_results=min(limit, 20),
            )

            web_results = []
            for i, result in enumerate(response.results or []):
                excerpts = result.excerpts or []
                web_results.append(
                    {
                        "url": result.url or "",
                        "title": result.title or "",
                        "description": " ".join(excerpts) if excerpts else "",
                        "position": i + 1,
                    }
                )

            return {"success": True, "data": {"web": web_results}}
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except ImportError as exc:
            return {
                "success": False,
                "error": f"Parallel SDK not installed: {exc}",
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Parallel search error: %s", exc)
            return {"success": False, "error": f"Parallel search failed: {exc}"}

    async def extract(
        self, urls: List[str], **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via the async SDK.

        Returns the legacy list-of-results shape that
        :func:`tools.web_tools.web_extract_tool` expects: one entry per
        successful URL plus one entry per failed URL with an ``error``
        field. Errors are not raised — they're returned as per-URL items.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {"url": u, "error": "Interrupted", "title": ""} for u in urls
                ]

            logger.info("Parallel extract: %d URL(s)", len(urls))
            client = await _get_parallel_client()
            response = await client.beta.extract(
                urls=urls,
                full_content=True,
            )

            results: List[Dict[str, Any]] = []
            for result in response.results or []:
                content = result.full_content or ""
                if not content:
                    content = "\n\n".join(result.excerpts or [])
                url = result.url or ""
                title = result.title or ""
                results.append(
                    {
                        "url": url,
                        "title": title,
                        "content": content,
                        "raw_content": content,
                        "metadata": {"sourceURL": url, "title": title},
                    }
                )

            for error in response.errors or []:
                results.append(
                    {
                        "url": error.url or "",
                        "title": "",
                        "content": "",
                        "error": error.content or error.error_type or "extraction failed",
                        "metadata": {"sourceURL": error.url or ""},
                    }
                )

            return results
        except ValueError as exc:
            return [{"url": u, "title": "", "content": "", "error": str(exc)} for u in urls]
        except ImportError as exc:
            return [
                {"url": u, "title": "", "content": "", "error": f"Parallel SDK not installed: {exc}"}
                for u in urls
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Parallel extract error: %s", exc)
            return [
                {"url": u, "title": "", "content": "", "error": f"Parallel extract failed: {exc}"}
                for u in urls
            ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Parallel",
            "badge": "paid",
            "tag": "Objective-tuned search + parallel page extraction.",
            "env_vars": [
                {
                    "key": "PARALLEL_API_KEY",
                    "prompt": "Parallel API key",
                    "url": "https://parallel.ai",
                },
            ],
        }
