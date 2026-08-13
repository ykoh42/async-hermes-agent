"""The native source loader is internal async infrastructure, not public API."""

from hermes_cli import async_source_loader


def test_source_loader_adds_no_public_callable_surface():
    owned_public_callables = sorted(
        name
        for name, value in vars(async_source_loader).items()
        if not name.startswith("_")
        and callable(value)
        and getattr(value, "__module__", None) == async_source_loader.__name__
    )

    assert owned_public_callables == []
    assert "__all__" not in vars(async_source_loader)
    assert not any(
        hasattr(async_source_loader, name)
        for name in (
            "locate_source_module",
            "load_source_module",
            "load_source_package",
            "unload_source_finder",
        )
    )
