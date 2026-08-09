"""Backend abstraction for Mem0 Platform and OSS modes."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import hashlib
import importlib
from typing import Any

import httpx

# Mirrors mem0ai 2.0.10's ``mem0.exceptions`` and ``client.utils`` behavior.
# Importing that package would also import its blocking client/OSS runtime, so
# the native HTTP backend keeps only the exception contract locally.

class MemoryError(Exception):
    """Structured Platform error matching the Mem0 SDK exception surface."""

    def __init__(
        self,
        message: str,
        error_code: str,
        *,
        details: dict[str, Any] | None = None,
        suggestion: str | None = None,
        debug_info: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.suggestion = suggestion
        self.debug_info = debug_info or {}
        super().__init__(message)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"error_code={self.error_code!r}, "
            f"details={self.details!r}, "
            f"suggestion={self.suggestion!r}, "
            f"debug_info={self.debug_info!r})"
        )


class AuthenticationError(MemoryError):
    pass


class RateLimitError(MemoryError):
    pass


class ValidationError(MemoryError):
    pass


class MemoryNotFoundError(MemoryError):
    pass


class NetworkError(MemoryError):
    pass


class MemoryQuotaExceededError(MemoryError):
    pass


_HTTP_ERROR_TYPES = {
    400: ValidationError,
    401: AuthenticationError,
    403: AuthenticationError,
    404: MemoryNotFoundError,
    408: NetworkError,
    409: ValidationError,
    413: MemoryQuotaExceededError,
    422: ValidationError,
    429: RateLimitError,
    500: MemoryError,
    502: NetworkError,
    503: NetworkError,
    504: NetworkError,
}

_HTTP_ERROR_SUGGESTIONS = {
    400: "Please check your request parameters and try again",
    401: "Please check your API key and authentication credentials",
    403: "You don't have permission to perform this operation",
    404: "The requested resource was not found",
    408: "Request timed out. Please try again",
    409: "Resource conflict. Please check your request",
    413: "Request too large. Please reduce the size of your request",
    422: "Invalid request data. Please check your input",
    429: "Rate limit exceeded. Please wait before making more requests",
    500: "Internal server error. Please try again later",
    502: "Service temporarily unavailable. Please try again later",
    503: "Service unavailable. Please try again later",
    504: "Gateway timeout. Please try again later",
}


def _platform_http_error(exc: httpx.HTTPStatusError) -> MemoryError:
    response = exc.response
    details: dict[str, Any] = {}
    message = response.text
    if response.headers.get("content-type", "").startswith("application/json"):
        try:
            decoded = response.json()
        except ValueError:
            decoded = None
        if isinstance(decoded, dict):
            details = decoded
            message = decoded.get("detail", message)
    debug_info: dict[str, Any] = {
        "status_code": response.status_code,
        "url": str(exc.request.url),
        "method": exc.request.method,
    }
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                debug_info["retry_after"] = int(retry_after)
            except ValueError:
                pass
        for header in (
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ):
            value = response.headers.get(header)
            if value:
                debug_info[header.lower().replace("-", "_")] = value
    error_type = _HTTP_ERROR_TYPES.get(response.status_code, MemoryError)
    return error_type(
        message or f"HTTP {response.status_code} error",
        f"HTTP_{response.status_code}",
        details=details,
        suggestion=_HTTP_ERROR_SUGGESTIONS.get(
            response.status_code, "Please try again later"
        ),
        debug_info=debug_info,
    )


def _platform_network_error(exc: httpx.RequestError) -> NetworkError:
    if isinstance(exc, httpx.TimeoutException):
        message = f"Request timed out: {exc}"
        code = "NET_TIMEOUT"
        error_type = "timeout"
    elif isinstance(exc, httpx.ConnectError):
        message = f"Connection failed: {exc}"
        code = "NET_CONNECT"
        error_type = "connection"
    else:
        message = f"Network request failed: {exc}"
        code = "NET_GENERIC"
        error_type = "request"
    return NetworkError(
        message,
        code,
        suggestion="Please check your internet connection and try again",
        debug_info={"error_type": error_type, "original_error": str(exc)},
    )


def _validate_search_query(query: str) -> str:
    if not isinstance(query, str):
        raise ValueError("Invalid query: must be a non-empty string.")
    trimmed = query.strip()
    if not trimmed:
        raise ValueError("Invalid query: cannot be empty or whitespace-only.")
    return trimmed


class Mem0Backend(ABC):
    """Unified interface over Platform (MemoryClient) and OSS (Memory) backends."""

    @abstractmethod
    async def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = False) -> list[dict]:
        ...

    @abstractmethod
    async def add(
        self,
        messages: list,
        *,
        user_id: str,
        agent_id: str,
        infer: bool = False,
        metadata: dict | None = None,
    ) -> dict:
        ...

    @abstractmethod
    async def update(self, memory_id: str, text: str) -> dict:
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> dict:
        ...

    async def close(self) -> None:
        """Release backend resources."""


def _unwrap_results(response: Any) -> list:
    """Normalize API response — extract results list from dict or pass through."""
    if isinstance(response, dict):
        return response.get("results", [])
    if isinstance(response, list):
        return response
    return []


class PlatformBackend(Mem0Backend):
    """Native-async client for the Mem0 Platform API."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "Mem0 API Key not provided. Please provide an API Key."
            )

        user_id = hashlib.md5(api_key.encode(), usedforsecurity=False).hexdigest()
        headers = {
            "Authorization": f"Token {api_key}",
            "Mem0-User-ID": user_id,
        }
        self._client = httpx.AsyncClient(
            base_url="https://api.mem0.ai",
            headers=headers,
            timeout=300.0,
        )
        self._initialize_lock = asyncio.Lock()
        self._initialized = False
        self.org_id: str | None = None
        self.project_id: str | None = None
        self.user_email: str | None = None

    async def _initialize(self) -> None:
        """Validate the API key at the same lifecycle boundary as the SDK."""
        if self._initialized:
            return
        async with self._initialize_lock:
            if self._initialized:
                return
            try:
                response = await self._client.get("/v1/ping/")
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as exc:
                try:
                    error_data = exc.response.json()
                    detail = error_data.get("detail", str(exc))
                except Exception:
                    detail = str(exc)
                raise ValueError(f"Error: {detail}") from exc
            if isinstance(data, dict):
                if data.get("org_id") and data.get("project_id"):
                    self.org_id = data.get("org_id")
                    self.project_id = data.get("project_id")
                self.user_email = data.get("user_email")
            self._initialized = True

    async def _json(self, method: str, path: str, **kwargs) -> Any:
        try:
            response = await self._client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json() if response.content else {}
        except httpx.HTTPStatusError as exc:
            raise _platform_http_error(exc) from exc
        except httpx.RequestError as exc:
            raise _platform_network_error(exc) from exc

    async def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = False) -> list[dict]:
        await self._initialize()
        query = _validate_search_query(query)
        response = await self._json(
            "POST",
            "/v3/memories/search/",
            json={
                "query": query,
                "filters": filters,
                "top_k": top_k,
                "rerank": rerank,
            },
        )
        return _unwrap_results(response)

    async def add(
        self,
        messages: list,
        *,
        user_id: str,
        agent_id: str,
        infer: bool = False,
        metadata: dict | None = None,
    ) -> dict:
        await self._initialize()
        body: dict[str, Any] = {
            "messages": messages,
            "user_id": user_id,
            "agent_id": agent_id,
            "infer": infer,
        }
        if metadata:
            body["metadata"] = metadata
        return await self._json("POST", "/v3/memories/add/", json=body)

    async def update(self, memory_id: str, text: str) -> dict:
        await self._initialize()
        await self._json(
            "PUT",
            f"/v1/memories/{memory_id}/",
            json={"text": text},
        )
        return {"result": "Memory updated.", "memory_id": memory_id}

    async def delete(self, memory_id: str) -> dict:
        await self._initialize()
        await self._json("DELETE", f"/v1/memories/{memory_id}/")
        return {"result": "Memory deleted.", "memory_id": memory_id}

    async def close(self) -> None:
        await self._client.aclose()


class SelfHostedBackend(Mem0Backend):
    """Direct HTTP backend for a self-hosted Mem0 server (the FastAPI ``server/``).

    mem0.MemoryClient can't be reused for self-hosted: it is hardwired to the
    cloud API — ``Authorization: Token`` auth and a ``GET /v1/ping/`` validation
    call in ``__init__`` that the self-hosted server does not expose (it would
    404 before any real request). This client talks to that server directly,
    using its actual contract: ``X-API-Key`` auth and the ``/memories`` /
    ``/search`` routes.
    """

    def __init__(self, api_key: str, host: str, transport=None):
        import httpx

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key  # omitted only for AUTH_DISABLED servers
        # Connect-level retries smooth over transient blips so a single
        # dropped SYN doesn't count toward the provider failure breaker.
        # ``transport`` is injectable for tests (httpx.MockTransport).
        if transport is None:
            transport = httpx.AsyncHTTPTransport(retries=2)
        self._client = httpx.AsyncClient(
            base_url=host.rstrip("/"), headers=headers, timeout=30.0,
            transport=transport,
        )

    async def _json(self, method: str, path: str, **kwargs) -> Any:
        resp = await self._client.request(method, path, **kwargs)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    async def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = False) -> list[dict]:
        # rerank is a platform-only feature; the self-hosted /search ignores it.
        body: dict[str, Any] = {"query": query, "top_k": top_k}
        if filters:
            body["filters"] = filters  # user_id belongs in filters (top-level is deprecated)
        return _unwrap_results(await self._json("POST", "/search", json=body))

    async def add(
        self,
        messages: list,
        *,
        user_id: str,
        agent_id: str,
        infer: bool = False,
        metadata: dict | None = None,
    ) -> dict:
        body: dict[str, Any] = {
            "messages": messages,
            "user_id": user_id,
            "agent_id": agent_id,
            "infer": infer,
        }
        if metadata:
            body["metadata"] = metadata
        return await self._json("POST", "/memories", json=body)

    async def update(self, memory_id: str, text: str) -> dict:
        await self._json("PUT", f"/memories/{memory_id}", json={"text": text})
        return {"result": "Memory updated.", "memory_id": memory_id}

    async def delete(self, memory_id: str) -> dict:
        await self._json("DELETE", f"/memories/{memory_id}")
        return {"result": "Memory deleted.", "memory_id": memory_id}

    async def close(self) -> None:
        try:
            await self._client.aclose()
        except Exception:
            pass


class OSSBackend(Mem0Backend):
    """Wraps mem0.Memory for self-hosted (OSS) mode."""

    def __init__(self, oss_config: dict):
        import os

        def _provider_block(name: str) -> dict:
            block = dict(oss_config[name])
            provider = str(block.get("provider") or "").strip().lower()
            provider_config = dict(block.get("config", {}))
            legacy_base = provider_config.pop("api_base", None)
            if legacy_base:
                from ._oss_providers import EMBEDDER_PROVIDERS, LLM_PROVIDERS

                provider_def = (
                    LLM_PROVIDERS if name == "llm" else EMBEDDER_PROVIDERS
                ).get(provider, {})
                canonical_key = provider_def.get("base_url_key")
                if canonical_key:
                    provider_config.setdefault(canonical_key, legacy_base)
            block["config"] = provider_config
            return block

        vector_store = dict(oss_config["vector_store"])
        vs_config = dict(vector_store.get("config", {}))

        if "path" in vs_config:
            vs_config["path"] = os.path.expanduser(vs_config["path"])

        embedder_config = oss_config.get("embedder", {}).get("config", {})
        dims = embedder_config.get("embedding_dims")
        if not dims:
            from ._oss_providers import KNOWN_DIMS
            model = embedder_config.get("model", "")
            dims = KNOWN_DIMS.get(model)
        if dims:
            vs_config["embedding_model_dims"] = dims
            self._collection_check: tuple[str, dict[str, Any], int] | None = (
                vector_store.get("provider", "qdrant"),
                vs_config,
                dims,
            )
        else:
            self._collection_check = None

        vector_store["config"] = vs_config

        config = {
            "vector_store": vector_store,
            "llm": _provider_block("llm"),
            "embedder": _provider_block("embedder"),
            "version": "v1.1",
        }
        self._config = config
        self._memory = None

    async def initialize(self) -> None:
        if self._memory is not None:
            return
        if self._collection_check is not None:
            await self._recreate_collection_if_dims_changed(
                *self._collection_check
            )
            self._collection_check = None
        from mem0 import AsyncMemory

        self._memory = AsyncMemory.from_config(self._config)

    @staticmethod
    async def _recreate_collection_if_dims_changed(provider: str, vs_config: dict, expected_dims: int) -> None:
        """Delete stale vector collection when embedding dimensions change."""
        collection_name = vs_config.get("collection_name", "mem0")
        if provider == "qdrant":
            try:
                from qdrant_client import AsyncQdrantClient
                path = vs_config.get("path")
                url = vs_config.get("url")
                if path:
                    client = AsyncQdrantClient(path=path)
                elif url:
                    client = AsyncQdrantClient(
                        url=url, api_key=vs_config.get("api_key")
                    )
                else:
                    return
                try:
                    if not await client.collection_exists(collection_name):
                        return
                    info = await client.get_collection(collection_name)
                    vectors = info.config.params.vectors
                    # Named-vector collections expose a dict; unnamed expose an object with .size.
                    if isinstance(vectors, dict):
                        first = next(iter(vectors.values()), None)
                        current_dims = first.size if first else None
                    else:
                        current_dims = getattr(vectors, "size", None)
                    if current_dims is not None and current_dims != expected_dims:
                        await client.delete_collection(collection_name)
                finally:
                    await client.close()
            except Exception:
                pass
        elif provider == "pgvector":
            try:
                psycopg = importlib.import_module("psycopg")
                pgsql = importlib.import_module("psycopg.sql")
                conn_params = {}
                for k in ("host", "port", "user", "password", "dbname"):
                    if vs_config.get(k):
                        conn_params[k] = vs_config[k]
                if vs_config.get("sslmode"):
                    conn_params["sslmode"] = vs_config["sslmode"]
                async with await psycopg.AsyncConnection.connect(
                    **conn_params,
                    autocommit=True,
                ) as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            "SELECT atttypmod FROM pg_attribute "
                            "WHERE attrelid = %s::regclass AND attname = 'vector'",
                            (collection_name,),
                        )
                        row = await cur.fetchone()
                        if row and row[0] > 0 and row[0] != expected_dims:
                            await cur.execute(
                                pgsql.SQL("DROP TABLE IF EXISTS {}").format(
                                    pgsql.Identifier(collection_name)
                                )
                            )
            except Exception:
                pass

    async def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = False) -> list[dict]:
        await self.initialize()
        memory = self._memory
        assert memory is not None
        response = await memory.search(query, filters=filters, top_k=top_k)
        return _unwrap_results(response)

    async def add(
        self,
        messages: list,
        *,
        user_id: str,
        agent_id: str,
        infer: bool = False,
        metadata: dict | None = None,
    ) -> dict:
        kwargs: dict[str, Any] = {"user_id": user_id, "agent_id": agent_id, "infer": infer}
        if metadata:
            kwargs["metadata"] = metadata
        await self.initialize()
        memory = self._memory
        assert memory is not None
        return await memory.add(messages, **kwargs)

    async def update(self, memory_id: str, text: str) -> dict:
        await self.initialize()
        memory = self._memory
        assert memory is not None
        await memory.update(memory_id, data=text)
        return {"result": "Memory updated.", "memory_id": memory_id}

    async def delete(self, memory_id: str) -> dict:
        await self.initialize()
        memory = self._memory
        assert memory is not None
        await memory.delete(memory_id)
        return {"result": "Memory deleted.", "memory_id": memory_id}

    async def close(self):
        if self._memory is None:
            return
        memory = self._memory
        self._memory = None
        vector_store = getattr(memory, "vector_store", None)
        client = getattr(vector_store, "client", None)
        aclose = getattr(client, "aclose", None)
        if callable(aclose):
            await aclose()
