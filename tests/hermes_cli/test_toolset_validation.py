from hermes_cli.toolset_validation import validate_platform_toolsets


def test_empty_or_non_mapping_has_no_warnings():
    assert validate_platform_toolsets(None, lambda _name: False) == []
    assert validate_platform_toolsets({}, lambda _name: False) == []


def test_unknown_toolset_is_reported_with_available_suggestion():
    valid = {"hermes-cli", "web"}
    warnings = validate_platform_toolsets(
        {"cli": ["hermes", "web"]},
        valid.__contains__,
    )
    assert any("unknown toolset 'hermes'" in warning for warning in warnings)
    assert any("did you mean 'hermes-cli'" in warning for warning in warnings)
    assert not any("zero valid" in warning for warning in warnings)


def test_nonempty_mapping_with_no_valid_toolsets_warns_loudly():
    warnings = validate_platform_toolsets(
        {"api": ["missing"]},
        lambda _name: False,
    )
    assert any("unknown toolset 'missing'" in warning for warning in warnings)
    assert any("zero valid toolsets" in warning for warning in warnings)
