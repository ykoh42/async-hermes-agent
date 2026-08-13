"""Side-effect-free validation for ``platform_toolsets`` configuration."""

from collections.abc import Callable


def validate_platform_toolsets(
    platform_toolsets: object,
    is_valid_toolset: Callable[[str], bool],
) -> list[str]:
    """Return warnings for unknown names or a zero-tool resolution."""
    warnings: list[str] = []
    if not isinstance(platform_toolsets, dict) or not platform_toolsets:
        return warnings

    valid_count = 0
    for platform, raw in platform_toolsets.items():
        names = raw if isinstance(raw, list) else [raw]
        for name in names:
            if not isinstance(name, str) or not name:
                continue
            if is_valid_toolset(name):
                valid_count += 1
                continue
            suggestion = f"hermes-{platform}"
            hint = (
                f" — did you mean '{suggestion}'?"
                if is_valid_toolset(suggestion)
                else ""
            )
            warnings.append(
                f"platform '{platform}' references unknown toolset "
                f"'{name}'{hint}"
            )

    if valid_count == 0:
        warnings.append(
            "platform_toolsets resolves to zero valid toolsets — the agent "
            "will have no tools."
        )
    return warnings
