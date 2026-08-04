"""Firecrawl web search + extract provider.

Subclasses :class:`agent.web_search_provider.WebSearchProvider` and uses
Firecrawl's HTTP API directly through :class:`httpx.AsyncClient`.

  - :func:`check_firecrawl_api_key` is re-exported by ``tools.web_tools``
    because the tools setup flow depends on that name.
  - :func:`_extract_web_search_results` / :func:`_extract_scrape_payload`
    normalize direct API response variants.
  - Per-URL extract loop with 60s timeout, redirect-aware SSRF re-check,
    website-policy gating, and format-aware content selection.

Config keys this provider responds to::

    web:
      search_backend: "firecrawl"     # explicit per-capability
      extract_backend: "firecrawl"    # explicit per-capability
      backend: "firecrawl"            # shared fallback (default)
Env vars::

    FIRECRAWL_API_KEY=...            # direct cloud auth
    FIRECRAWL_API_URL=...            # self-hosted Firecrawl
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from agent.web_search_provider import WebSearchProvider
from tools.url_safety import is_safe_url
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)


def _get_direct_firecrawl_config() -> Optional[Dict[str, str]]:
    """Return direct Firecrawl request configuration, or ``None`` when unset."""
    from hermes_cli.config import get_env_value

    api_key = (get_env_value("FIRECRAWL_API_KEY") or "").strip()
    api_url = (get_env_value("FIRECRAWL_API_URL") or "").strip().rstrip("/")

    if not api_key and not api_url:
        return None

    kwargs: Dict[str, str] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if api_url:
        kwargs["api_url"] = api_url

    return kwargs


def _has_direct_firecrawl_config() -> bool:
    """Return True when direct Firecrawl config is explicitly configured."""
    return _get_direct_firecrawl_config() is not None


def check_firecrawl_api_key() -> bool:
    """Return True when a direct or self-hosted Firecrawl backend is usable.

    Re-exported by :mod:`tools.web_tools` for backward compatibility with
    existing tests and the ``hermes tools`` setup flow.
    """
    return _has_direct_firecrawl_config()


def _raise_web_backend_configuration_error() -> None:
    """Raise a clear error for unsupported web backend configuration."""
    message = (
        "Web tools are not configured. "
        "Set FIRECRAWL_API_KEY for cloud Firecrawl or set FIRECRAWL_API_URL "
        "for a self-hosted Firecrawl instance."
    )
    raise ValueError(message)


# ---------------------------------------------------------------------------
# Response shape normalization
# ---------------------------------------------------------------------------


def _to_plain_object(value: Any) -> Any:
    """Convert SDK objects to plain python data structures when possible."""
    if value is None:
        return None

    if isinstance(value, (dict, list, str, int, float, bool)):
        return value

    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:  # noqa: BLE001
            pass

    if hasattr(value, "__dict__"):
        try:
            return {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
        except Exception:  # noqa: BLE001
            pass

    return value


def _normalize_result_list(values: Any) -> List[Dict[str, Any]]:
    """Normalize mixed list payloads into a list of dictionaries."""
    if not isinstance(values, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in values:
        plain = _to_plain_object(item)
        if isinstance(plain, dict):
            normalized.append(plain)
    return normalized


def _extract_web_search_results(response: Any) -> List[Dict[str, Any]]:
    """Extract Firecrawl search results across direct API response shapes."""
    response_plain = _to_plain_object(response)

    if isinstance(response_plain, dict):
        data = response_plain.get("data")
        if isinstance(data, list):
            return _normalize_result_list(data)

        if isinstance(data, dict):
            data_web = _normalize_result_list(data.get("web"))
            if data_web:
                return data_web
            data_results = _normalize_result_list(data.get("results"))
            if data_results:
                return data_results

        top_web = _normalize_result_list(response_plain.get("web"))
        if top_web:
            return top_web

        top_results = _normalize_result_list(response_plain.get("results"))
        if top_results:
            return top_results

    if hasattr(response, "web"):
        return _normalize_result_list(getattr(response, "web", []))

    return []


def _extract_scrape_payload(scrape_result: Any) -> Dict[str, Any]:
    """Normalize Firecrawl scrape payload shape across direct API variants."""
    result_plain = _to_plain_object(scrape_result)
    if not isinstance(result_plain, dict):
        return {}

    nested = result_plain.get("data")
    if isinstance(nested, dict):
        return nested

    return result_plain


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class FirecrawlWebSearchProvider(WebSearchProvider):
    """Native-async Firecrawl search + extract provider."""

    @staticmethod
    def _request_config() -> tuple[str, str]:
        """Resolve the active Firecrawl HTTP origin and bearer token."""
        direct_config = _get_direct_firecrawl_config()
        if direct_config is None:
            _raise_web_backend_configuration_error()
        return (
            (direct_config.get("api_url") or "https://api.firecrawl.dev/v1").rstrip("/"),
            direct_config.get("api_key", ""),
        )

    @classmethod
    async def _request(cls, endpoint: str, payload: Dict[str, Any]) -> Any:
        import httpx

        origin, token = cls._request_config()
        if not origin.endswith("/v1"):
            origin = f"{origin}/v1"
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{origin}/{endpoint.lstrip('/')}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "User-Agent": "Hermes-Agent",
                },
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    @property
    def name(self) -> str:
        return "firecrawl"

    @property
    def display_name(self) -> str:
        return "Firecrawl"

    def is_available(self) -> bool:
        """Return True when direct or self-hosted Firecrawl is configured."""
        return check_firecrawl_api_key()

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    async def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a Firecrawl search.

        Normalizes the direct HTTP response via
        :func:`_extract_web_search_results`. In-flight errors are surfaced as
        ``{"success": False, "error": ...}``.
        """
        from tools.interrupt import is_interrupted

        if is_interrupted():
            return {"success": False, "error": "Interrupted"}

        logger.info("Firecrawl search: '%s' (limit=%d)", query, limit)
        try:
            response = await self._request(
                "search",
                {"query": query, "limit": max(1, min(int(limit), 100))},
            )
            web_results = _extract_web_search_results(response)
            logger.info("Firecrawl: found %d search results", len(web_results))
            return {"success": True, "data": {"web": web_results}}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Firecrawl search error: %s", exc)
            return {"success": False, "error": f"Firecrawl search failed: {exc}"}

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via Firecrawl.

        Async; each URL is scraped through the native HTTP API with a 60s
        timeout. After scraping, the final URL (post-redirect) is
        re-checked against website-access policy.

        Accepted kwargs (others ignored for forward compat):
          - ``format``: ``"markdown"`` or ``"html"``; default is both
            (request both, return markdown when available).

        Returns the legacy per-URL list-of-results shape. Per-URL failures
        (timeout, SSRF block, scrape error, policy block) become items
        with an ``error`` field rather than raising.
        """
        from tools.interrupt import is_interrupted as _is_interrupted

        if _is_interrupted():
            return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]

        format = kwargs.get("format")
        formats: List[str] = []
        if format == "markdown":
            formats = ["markdown"]
        elif format == "html":
            formats = ["html"]
        else:
            formats = ["markdown", "html"]

        # check_website_access is the legacy policy gate; imported at
        # module level (lazy-friendly because the website_policy import is
        # cheap) so monkeypatching it in tests works as expected.

        results: List[Dict[str, Any]] = []

        for url in urls:
            if _is_interrupted():
                results.append({"url": url, "error": "Interrupted", "title": ""})
                continue

            # Pre-scrape website policy gate
            blocked = await check_website_access(url)
            if blocked:
                logger.info(
                    "Blocked web_extract for %s by rule %s",
                    blocked["host"],
                    blocked["rule"],
                )
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "error": blocked["message"],
                        "blocked_by_policy": {
                            "host": blocked["host"],
                            "rule": blocked["rule"],
                            "source": blocked["source"],
                        },
                    }
                )
                continue

            try:
                logger.info("Firecrawl scraping: %s", url)
                try:
                    scrape_result = await asyncio.wait_for(
                        self._request(
                            "scrape",
                            {"url": url, "formats": formats},
                        ),
                        timeout=60,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Firecrawl scrape timed out for %s", url)
                    results.append(
                        {
                            "url": url,
                            "title": "",
                            "content": "",
                            "error": (
                                "Scrape timed out after 60s — page may be too large "
                                "or unresponsive. Try browser_navigate instead."
                            ),
                        }
                    )
                    continue

                scrape_payload = _extract_scrape_payload(scrape_result)
                metadata = scrape_payload.get("metadata", {})
                content_markdown = scrape_payload.get("markdown")
                content_html = scrape_payload.get("html")

                # Ensure metadata is a dict (SDK may return a typed object)
                if not isinstance(metadata, dict):
                    if hasattr(metadata, "model_dump"):
                        metadata = metadata.model_dump()
                    elif hasattr(metadata, "__dict__"):
                        metadata = metadata.__dict__
                    else:
                        metadata = {}

                title = metadata.get("title", "")
                final_url = metadata.get("sourceURL", url)

                # Re-check SSRF safety after any redirect reported by Firecrawl.
                if not await is_safe_url(final_url):
                    logger.info(
                        "Blocked redirected web_extract for unsafe final URL: %s",
                        final_url,
                    )
                    results.append(
                        {
                            "url": final_url,
                            "title": title,
                            "content": "",
                            "raw_content": "",
                            "error": (
                                "Blocked: URL targets a private or internal "
                                "network address"
                            ),
                        }
                    )
                    continue

                # Re-check website-access policy after any redirect
                final_blocked = await check_website_access(final_url)
                if final_blocked:
                    logger.info(
                        "Blocked redirected web_extract for %s by rule %s",
                        final_blocked["host"],
                        final_blocked["rule"],
                    )
                    results.append(
                        {
                            "url": final_url,
                            "title": title,
                            "content": "",
                            "raw_content": "",
                            "error": final_blocked["message"],
                            "blocked_by_policy": {
                                "host": final_blocked["host"],
                                "rule": final_blocked["rule"],
                                "source": final_blocked["source"],
                            },
                        }
                    )
                    continue

                # Choose markdown vs html according to the requested format
                if format == "markdown" or (format is None and content_markdown):
                    chosen_content = content_markdown
                else:
                    chosen_content = content_html or content_markdown or ""

                results.append(
                    {
                        "url": final_url,
                        "title": title,
                        "content": chosen_content,
                        "raw_content": chosen_content,
                        "metadata": metadata,
                    }
                )
            except Exception as scrape_err:  # noqa: BLE001
                logger.debug("Firecrawl scrape failed for %s: %s", url, scrape_err)
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": str(scrape_err),
                    }
                )

        return results

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Firecrawl",
            "badge": "paid · self-hostable",
            "tag": "Full search + extract via direct or self-hosted Firecrawl.",
            "env_vars": [
                {
                    "key": "FIRECRAWL_API_KEY",
                    "prompt": "Firecrawl API key (or leave blank for self-hosted)",
                    "url": "https://docs.firecrawl.dev/introduction",
                },
            ],
        }
