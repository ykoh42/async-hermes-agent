import pytest

from hermes_cli import lifecycle, plugins


@pytest.mark.asyncio
async def test_dispatch_invokes_native_plugin_hooks(monkeypatch):
    async def invoke(name, **kwargs):
        return [name, kwargs]

    monkeypatch.setattr(plugins, "invoke_hook", invoke)
    assert await lifecycle.invoke_hook("custom", value=1) == [
        "custom",
        {"value": 1},
    ]
