"""Retained delegation defaults from upstream v2026.8.16."""


def test_delegation_defaults_match_upstream():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from tools.delegate_tool import (
        DEFAULT_MAX_ITERATIONS,
        _DEFAULT_MAX_CONCURRENT_CHILDREN,
    )

    delegation = DEFAULT_CONFIG["delegation"]
    assert delegation["max_iterations"] == DEFAULT_MAX_ITERATIONS == 250
    assert (
        delegation["max_concurrent_children"]
        == _DEFAULT_MAX_CONCURRENT_CHILDREN
        == 10
    )
