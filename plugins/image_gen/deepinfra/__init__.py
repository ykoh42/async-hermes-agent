"""DeepInfra image generation backend.

Exposes DeepInfra's image-gen catalog (FLUX, Qwen-Image-Edit, …) through
the OpenAI-compatible ``/v1/openai/images/generations`` endpoint as an
:class:`ImageGenProvider` implementation.

**Fully dynamic model discovery.** Unlike the other image-gen plugins in
this tree (which ship a hardcoded ``_MODELS`` dict), DeepInfra publishes
a single tagged catalog at
``https://api.deepinfra.com/v1/openai/models?filter=true&sort_by=hermes``
where each entry's ``metadata.tags`` declares its surface (``image-gen``
here). ``list_models()`` filters that catalog via
:func:`hermes_cli.models._fetch_deepinfra_models_by_tag` so newly added
models show up in ``hermes tools`` automatically. No model ids are
hardcoded in this file — if a model is retired upstream, it disappears
from hermes the next time the catalog is fetched, no patch required.

Model selection (first hit wins):

1. ``DEEPINFRA_IMAGE_MODEL`` env var
2. ``image_gen.deepinfra.model`` in ``config.yaml``
3. First model from the live catalog

When all three are absent (catalog unreachable, nothing configured),
``generate()`` returns an :func:`error_response` rather than guessing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agent.secret_scope import get_secret
from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    _close_owned_client,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    save_url_image,
    success_response,
)

logger = logging.getLogger(__name__)


async def _close_client_after_request(
    client: Any,
    cancellation: asyncio.CancelledError | None,
) -> None:
    """Finish client close without replacing the request's cancellation."""
    try:
        await _close_owned_client(client)
    except asyncio.CancelledError:  # noqa: ASYNC103 - request cancellation wins below
        if cancellation is None:
            raise
    except Exception as exc:
        if cancellation is not None:
            raise cancellation from exc
        raise
    if cancellation is not None:
        raise cancellation


# DeepInfra accepts standard OpenAI ``size`` strings. Mirrors the
# OpenAI plugin's mapping so aspect_ratio semantics stay consistent
# across the agent's image_generate tool surface.
_SIZES = {
    "landscape": "1536x1024",
    "square": "1024x1024",
    "portrait": "1024x1536",
}


async def _load_deepinfra_image_config() -> dict[str, Any]:
    """Read ``image_gen.deepinfra`` from config.yaml."""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = await load_config_readonly()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        di_section = section.get("deepinfra") if isinstance(section, dict) else None
        return di_section if isinstance(di_section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen.deepinfra config: %s", exc)
        return {}


async def _live_models() -> list[dict[str, Any]] | None:
    """Fetch ``image-gen``-tagged models from the DeepInfra catalog."""
    try:
        from hermes_cli.models import _fetch_deepinfra_models_by_tag
    except Exception as exc:
        logger.debug("Cannot import _fetch_deepinfra_models_by_tag: %s", exc)
        return None
    return await _fetch_deepinfra_models_by_tag("image-gen")


def _format_catalog_row(item: dict[str, Any]) -> dict[str, Any]:
    """Format a catalog item into the picker row shape."""
    mid = item.get("id", "")
    metadata = item.get("metadata") or {}
    pricing = metadata.get("pricing") if isinstance(metadata, dict) else None
    price = ""
    if isinstance(pricing, dict) and pricing.get("per_image_unit") is not None:
        try:
            price = f"${float(pricing['per_image_unit']):.4f}/image"
        except (TypeError, ValueError):
            price = ""
    row: dict[str, Any] = {
        "id": mid,
        "display": mid.split("/", 1)[-1] if "/" in mid else mid,
        "strengths": metadata.get("description", "") if isinstance(metadata, dict) else "",
    }
    if price:
        row["price"] = price
    if isinstance(metadata, dict):
        for key in ("default_width", "default_height", "default_iterations"):
            if metadata.get(key) is not None:
                row[key] = metadata[key]
    return row


def _resolve_model(catalog: list[dict[str, Any]], cfg: dict[str, Any]) -> str | None:
    """Pick the model id (env > config > first live result, else None).

    Takes the already-loaded ``image_gen.deepinfra`` config so ``generate()``
    reads config once instead of via a second ``load_config`` deepcopy.
    """
    env_override = (get_secret("DEEPINFRA_IMAGE_MODEL", "") or "").strip()
    if env_override:
        return env_override
    cfg_model = cfg.get("model") if isinstance(cfg, dict) else None
    if isinstance(cfg_model, str) and cfg_model.strip():
        return cfg_model.strip()
    if catalog:
        first = catalog[0].get("id")
        if isinstance(first, str) and first:
            return first
    return None


class DeepInfraImageGenProvider(ImageGenProvider):
    """DeepInfra ``images.generations`` backend.

    Catalog is discovered live from the DeepInfra ``/models`` endpoint
    filtered by the ``image-gen`` surface tag.
    """

    @property
    def name(self) -> str:
        return "deepinfra"

    @property
    def display_name(self) -> str:
        return "DeepInfra"

    async def is_available(self) -> bool:
        return bool((get_secret("DEEPINFRA_API_KEY", "") or "").strip())

    async def list_models(self) -> list[dict[str, Any]]:
        live = await _live_models()
        if not live:
            return []
        return [_format_catalog_row(item) for item in live]

    async def default_model(self) -> str | None:
        rows = await self.list_models()
        if rows:
            return rows[0].get("id")
        return None

    async def capabilities(self) -> dict[str, Any]:
        """DeepInfra's OpenAI-compatible generation surface is text-only."""
        return {"modalities": ["text"], "max_reference_images": 0}

    async def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": "DeepInfra",
            "badge": "paid",
            "tag": "FLUX, Qwen-Image, … — live catalog from api.deepinfra.com",
            "env_vars": [
                {
                    "key": "DEEPINFRA_API_KEY",
                    "prompt": "DeepInfra API key",
                    "url": "https://deepinfra.com/dash/api_keys",
                },
            ],
        }

    async def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if kwargs.get("image_url") or kwargs.get("reference_image_urls"):
            return error_response(
                error=(
                    "DeepInfra image generation is text-to-image only in this "
                    "backend; image_url and reference_image_urls are unsupported."
                ),
                error_type="modality_unsupported",
                provider="deepinfra",
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="deepinfra",
                aspect_ratio=aspect,
            )

        api_key = (get_secret("DEEPINFRA_API_KEY", "") or "").strip()
        if not api_key:
            return error_response(
                error=(
                    "DEEPINFRA_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → DeepInfra to configure, or `hermes setup` "
                    "to add the key."
                ),
                error_type="auth_required",
                provider="deepinfra",
                aspect_ratio=aspect,
            )

        di_cfg = await _load_deepinfra_image_config()
        catalog = await _live_models() or []
        model_id = _resolve_model(catalog, di_cfg)
        if not model_id:
            return error_response(
                error=(
                    "No DeepInfra image-gen model available. Pin one in "
                    "config.yaml under image_gen.deepinfra.model, set "
                    "DEEPINFRA_IMAGE_MODEL, or check connectivity to "
                    "api.deepinfra.com so the live catalog can be fetched."
                ),
                error_type="no_model_available",
                provider="deepinfra",
                prompt=prompt,
                aspect_ratio=aspect,
            )
        size = _SIZES.get(aspect, _SIZES["square"])
        from hermes_cli.models import deepinfra_base_url
        base_url = deepinfra_base_url(di_cfg)

        # DeepInfra's /images/generations is OpenAI-compatible — use the
        # openai SDK so we inherit its retry, timeout, and error mapping
        # (mirrors the existing OpenAI image-gen plugin).
        try:
            import openai
        except ImportError:
            return error_response(
                error="openai Python package not installed (pip install openai)",
                error_type="missing_dependency",
                provider="deepinfra",
                aspect_ratio=aspect,
            )

        from agent.ssl_verify import _create_openai_sdk_client

        client = await _create_openai_sdk_client(
            openai.AsyncOpenAI,
            api_key=api_key,
            base_url=base_url,
        )
        cancellation: asyncio.CancelledError | None = None
        try:
            response = await client.images.generate(
                model=model_id,
                prompt=prompt,
                size=size,  # type: ignore[arg-type]  # _SIZES values are valid OpenAI image sizes
                n=1,
            )
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - closed below
            cancellation = exc
        except Exception as exc:
            logger.debug("DeepInfra image generation failed", exc_info=True)
            return error_response(
                error=f"DeepInfra image generation failed: {exc}",
                error_type="api_error",
                provider="deepinfra",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        finally:
            await _close_client_after_request(client, cancellation)

        data = getattr(response, "data", None) or []
        if not data:
            return error_response(
                error="DeepInfra returned no image data",
                error_type="empty_response",
                provider="deepinfra",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first = data[0]
        b64 = getattr(first, "b64_json", None)
        url = getattr(first, "url", None)

        # Drop the ``vendor/`` prefix and any colons so the saved filename
        # stays a single path component on every OS.
        short = model_id.split("/", 1)[-1].replace(":", "_")

        if b64:
            try:
                saved_path = await save_b64_image(
                    b64,
                    prefix=f"deepinfra_{short}",
                )
            except Exception as exc:
                return error_response(
                    error=f"Could not save image to cache: {exc}",
                    error_type="io_error",
                    provider="deepinfra",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            image_ref = str(saved_path)
        elif url:
            # Materialise the (often short-lived) delivery URL locally so a
            # downstream consumer (Telegram send_photo, browser fetch) doesn't
            # get a dead link — mirrors the openai/xai/krea image plugins.
            # Best-effort: fall back to the bare URL if the download fails.
            try:
                image_ref = str(
                    await save_url_image(url, prefix=f"deepinfra_{short}")
                )
            except Exception as exc:
                logger.debug("DeepInfra: caching delivery URL failed (%s); returning URL", exc)
                image_ref = url
        else:
            return error_response(
                error="DeepInfra response contained neither b64_json nor URL",
                error_type="empty_response",
                provider="deepinfra",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="deepinfra",
            extra={"size": size},
        )


def register(ctx) -> None:
    """Plugin entry point — wire ``DeepInfraImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(DeepInfraImageGenProvider())
