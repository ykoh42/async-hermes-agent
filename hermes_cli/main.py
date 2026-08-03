#!/usr/bin/env python3
"""Slim Hermes command entry point.

The training runtime deliberately has no interactive UI, gateway, scheduler,
or tool-installation command surface.  Agent execution lives in
``run_agent.py`` and ``batch_runner.py``; this entry point is retained at its
upstream path solely to configure external MCP servers.
"""

import hermes_bootstrap  # noqa: F401

import argparse
import os
from importlib.metadata import PackageNotFoundError, version


# Kept for plugin-loader compatibility.  The slim runtime does not register
# auxiliary UI tasks through the CLI entry point.
_AUX_TASKS: tuple[str, ...] = ()


def _configure_runtime(args: argparse.Namespace) -> None:
    if args.safe_mode:
        os.environ["HERMES_SAFE_MODE"] = "1"
    if args.ignore_user_config:
        os.environ["HERMES_IGNORE_USER_CONFIG"] = "1"


def cmd_mcp(args: argparse.Namespace) -> None:
    from hermes_cli.mcp_config import mcp_command

    mcp_command(args)


def build_mcp_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the retained external-MCP management surface."""
    parser = subparsers.add_parser("mcp", help="Configure external MCP servers")
    actions = parser.add_subparsers(dest="mcp_action")

    add = actions.add_parser("add", help="Add an MCP server")
    add.add_argument("name")
    transport = add.add_mutually_exclusive_group()
    transport.add_argument("--url", help="Streamable HTTP MCP endpoint")
    transport.add_argument("--command", dest="mcp_command", help="Stdio MCP command")
    transport.add_argument("--preset", help="Known MCP preset")
    add.add_argument(
        "--args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments for --command (must be last)",
    )
    add.add_argument("--env", action="append", default=[], help="Stdio environment KEY=VALUE")
    add.add_argument("--auth", choices=("oauth", "header"))
    add.add_argument("--connect-timeout", type=float)

    remove = actions.add_parser("remove", aliases=["rm"], help="Remove an MCP server")
    remove.add_argument("name")
    actions.add_parser("list", aliases=["ls"], help="List configured MCP servers")
    test = actions.add_parser("test", help="Test an MCP server connection")
    test.add_argument("name")
    login = actions.add_parser("login", help="Authenticate an OAuth MCP server")
    login.add_argument("name")
    reauth = actions.add_parser("reauth", help="Re-authenticate OAuth MCP servers")
    reauth.add_argument("name", nargs="?")
    reauth.add_argument("--all", action="store_true")
    parser.set_defaults(func=cmd_mcp)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes",
        description="Hermes training runtime — external MCP configuration.",
    )
    parser.add_argument("--version", action="store_true", help="Print package version")
    parser.add_argument("--safe-mode", action="store_true", help="Disable unsafe runtime extensions")
    parser.add_argument(
        "--ignore-user-config", action="store_true", help="Ignore user configuration"
    )
    subparsers = parser.add_subparsers(dest="command")
    build_mcp_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_runtime(args)

    if args.version:
        try:
            print(version("async-hermes-agent"))
        except PackageNotFoundError:
            print("async-hermes-agent (source checkout)")
        return 0
    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    result = args.func(args)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
