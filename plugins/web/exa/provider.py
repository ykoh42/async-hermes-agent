"""Exa web search + content extraction — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. The active
provider uses Exa's native async HTTP API; the optional SDK is imported only
by the legacy client-inspection helper.

Config keys this provider responds to::

    web:
      search_backend: "exa"      # explicit per-capability
      extract_backend: "exa"     # explicit per-capability
      backend: "exa"             # shared fallback for both

Env var::

    EXA_API_KEY=...    # https://exa.ai (paid tier; free trial available)

The previous in-tree implementation lived at
``tools.web_tools._exa_search`` / ``_exa_extract``; this file is the
canonical replacement. Behavior is bit-for-bit identical aside from the
ABC method-name change.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

# Module-level note: the canonical ``_exa_client`` cache slot lives on
# :mod:`tools.web_tools` so tests that do ``tools.web_tools._exa_client =
# None`` between cases see fresh state. The plugin reads/writes through
# that public module (see :func:`_get_exa_client`).


def _get_exa_client() -> Any:
    """Lazy-import and cache an Exa SDK client.

    Cache lives on :mod:`tools.web_tools` (as ``_exa_client``) so unit
    tests that reset that name between cases keep working. Raises
    ``ValueError`` when ``EXA_API_KEY`` is unset.
    """
    import tools.web_tools as _wt

    cached = getattr(_wt, "_exa_client", None)
    if cached is not None:
        return cached

    from agent.web_search_provider import get_provider_env

    api_key = get_provider_env("EXA_API_KEY")
    if not api_key:
        raise ValueError(
            "EXA_API_KEY environment variable not set. "
            "Get your API key at https://exa.ai"
        )

    try:
        from exa_py import Exa  # noqa: WPS433 — deliberately lazy
    except ImportError as exc:
        raise ImportError(
            "The optional Exa SDK is not installed. Install async-hermes-agent[exa]."
        ) from exc

    client = Exa(api_key=api_key)
    client.headers["x-exa-integration"] = "hermes-agent"
    _wt._exa_client = client
    return client


def _reset_client_for_tests() -> None:
    """Drop the cached Exa client so tests can re-instantiate cleanly."""
    import tools.web_tools as _wt

    _wt._exa_client = None


class ExaWebSearchProvider(WebSearchProvider):
    """Exa search + extract provider.

    The provider uses Exa's JSON HTTP API directly.  This keeps both
    operations native async even when the optional ``exa-py`` SDK is not
    installed; the SDK remains available through ``_get_exa_client`` for
    legacy introspection only and is never used on the active path.
    """

    @staticmethod
    async def _request(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        from agent.web_search_provider import get_provider_env

        api_key = get_provider_env("EXA_API_KEY")
        if not api_key:
            raise ValueError(
                "EXA_API_KEY environment variable not set. "
                "Get your API key at https://exa.ai"
            )
        import httpx

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"https://api.exa.ai/{endpoint.lstrip('/')}",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                    "User-Agent": "Hermes-Agent",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Exa API returned a non-object response")
            return data

    @property
    def name(self) -> str:
        return "exa"

    @property
    def display_name(self) -> str:
        return "Exa"

    def is_available(self) -> bool:
        """Return True when ``EXA_API_KEY`` is set to a non-empty value."""
        from agent.web_search_provider import get_provider_env

        return bool(get_provider_env("EXA_API_KEY"))

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    async def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute an Exa search.

        Returns ``{"success": True, "data": {"web": [{...}, ...]}}`` on
        success, ``{"success": False, "error": str}`` on failure (incl.
        missing API key and SDK install errors).
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            logger.info("Exa search: '%s' (limit=%d)", query, limit)
            response = await self._request(
                "search",
                {
                    "query": query,
                    "numResults": max(1, min(int(limit), 100)),
                    "contents": {"highlights": {"maxCharacters": 1000}},
                },
            )

            web_results = []
            for i, result in enumerate(response.get("results") or []):
                highlights = result.get("highlights") or []
                web_results.append(
                    {
                        "url": result.get("url") or "",
                        "title": result.get("title") or "",
                        "description": " ".join(highlights) if highlights else "",
                        "position": i + 1,
                    }
                )

            return {"success": True, "data": {"web": web_results}}
        except ValueError as exc:
            # Raised by _get_exa_client when EXA_API_KEY missing
            return {"success": False, "error": str(exc)}
        except ImportError as exc:
            return {"success": False, "error": f"Exa SDK not installed: {exc}"}
        except Exception as exc:  # noqa: BLE001 — surface as failure
            logger.warning("Exa search error: %s", exc)
            return {"success": False, "error": f"Exa search failed: {exc}"}

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via Exa.

        Returns a list of result dicts shaped for the legacy LLM
        post-processing pipeline. On per-URL or whole-batch failure,
        results carry an ``error`` field rather than raising.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {"url": u, "error": "Interrupted", "title": ""} for u in urls
                ]

            logger.info("Exa extract: %d URL(s)", len(urls))
            response = await self._request(
                "contents",
                {"ids": urls, "text": True},
            )

            results: List[Dict[str, Any]] = []
            for result in response.get("results") or []:
                content = result.get("text") or ""
                url = result.get("url") or ""
                title = result.get("title") or ""
                results.append(
                    {
                        "url": url,
                        "title": title,
                        "content": content,
                        "raw_content": content,
                        "metadata": {"sourceURL": url, "title": title},
                    }
                )
            return results
        except ValueError as exc:
            return [{"url": u, "title": "", "content": "", "error": str(exc)} for u in urls]
        except ImportError as exc:
            return [
                {"url": u, "title": "", "content": "", "error": f"Exa SDK not installed: {exc}"}
                for u in urls
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Exa extract error: %s", exc)
            return [
                {"url": u, "title": "", "content": "", "error": f"Exa extract failed: {exc}"}
                for u in urls
            ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Exa",
            "badge": "paid",
            "tag": "Semantic + neural web search with content extraction.",
            "env_vars": [
                {
                    "key": "EXA_API_KEY",
                    "prompt": "Exa API key",
                    "url": "https://exa.ai",
                },
            ],
        }
