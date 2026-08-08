"""Tests for the unconditional hardline command blocklist.

The hardline list is a floor below yolo: a small set of commands so
catastrophic they should never run via the agent, regardless of --yolo,
gateway /yolo, approvals.mode=off, or cron approve mode.

Inspired by Mercury Agent's permission-hardened blocklist.
"""

import pytest

from tools.approval import (
    HARDLINE_PATTERNS,
    _check_sudo_stdin_guard,
    detect_hardline_command,
    check_all_command_guards,
)
# -------------------------------------------------------------------------

# Commands that MUST be hardline-blocked.
_HARDLINE_BLOCK = [
    # rm -rf targeting root / system dirs / home
    "rm -rf /",
    "rm -rf /*",
    # Shell-equivalent spellings of "rm -rf /": repeated slashes and
    # current/parent-dir segments all collapse back to root, so they must
    # hit the hardline floor too (regression: these used to slip through the
    # root pattern's target group and fall to the softer DANGEROUS_PATTERNS
    # rule, which --yolo / approvals.mode=off / cron approve-mode bypass).
    "rm -rf //",
    "rm -rf /.",
    "rm -rf /./",
    "rm -rf /..",
    "rm -rf //*",
    "rm -fr /./",
    "ls && rm -rf //",
    "rm -rf /home",
    "rm -rf /home/*",
    "rm -rf /etc",
    "rm -rf /usr",
    "rm -rf /var",
    "rm -rf /boot",
    "rm -rf /bin",
    "rm --recursive --force /",
    "rm -fr /",
    "sudo rm -rf /",
    "rm -rf ~",
    "rm -rf ~/",
    "rm -rf ~/*",
    "rm -rf $HOME",
    # Quoted path idioms — the recommended shell form for paths with special
    # chars. These previously slipped past the floor because the surrounding
    # quote broke both the flag group and the (\s|$) terminator (regression
    # guard: catastrophic disk/home wipe under --yolo / approvals.mode=off).
    'rm -rf "/"',
    "rm -rf '/'",
    'rm -rf "/*"',
    'rm -rf "/etc"',
    "rm -rf '/etc'",
    'rm -rf "/home"',
    'rm -rf "/usr"',
    'rm -rf "$HOME"',
    "rm -rf '$HOME'",
    'rm -rf "$HOME/"',
    'rm -rf "~"',
    'sudo rm -rf "/"',
    'rm -rf "/" && echo done',
    # ${HOME} brace form (universally common, previously unmatched).
    "rm -rf ${HOME}",
    'rm -rf "${HOME}"',
    "rm -fr ${HOME}",
    # Filesystem format
    "mkfs.ext4 /dev/sda1",
    "mkfs /dev/sdb",
    "mkfs.xfs /dev/nvme0n1",
    # Raw block device overwrites
    "dd if=/dev/zero of=/dev/sda bs=1M",
    "dd if=/dev/urandom of=/dev/nvme0n1",
    "dd if=anything of=/dev/hda",
    "echo bad > /dev/sda",
    "cat /dev/urandom > /dev/sdb",
    # Fork bomb
    ":(){ :|:& };:",
    # System-wide kill
    "kill -9 -1",
    "kill -1",
    # Shutdown / reboot / halt
    "shutdown -h now",
    "shutdown -r now",
    "sudo shutdown now",
    "reboot",
    "sudo reboot",
    "halt",
    "poweroff",
    "init 0",
    "init 6",
    "telinit 0",
    "systemctl poweroff",
    "systemctl reboot",
    "systemctl halt",
    # Compound / subshell variants
    "ls; reboot",
    "echo done && shutdown -h now",
    "false || halt",
    "$(reboot)",
    "`shutdown now`",
    "sudo -E shutdown now",
    "env FOO=1 reboot",
    "exec shutdown",
    "nohup reboot",
    "setsid poweroff",
    # Bare subshell `(cmd)` and brace-group `{ cmd; }` openers put the trigger
    # at a real command position, so they must hit the floor just like `$(…)`.
    # These slipped through before the quote-aware command-start tokenizer
    # learned to recognize `(` / `{` (issue: (reboot) walked past --yolo).
    "(reboot)",
    "( reboot )",
    "(shutdown -h now)",
    "(poweroff)",
    "(halt)",
    "(init 0)",
    "(systemctl reboot)",
    "(sudo reboot)",
    "{ reboot; }",
    "{ shutdown -h now; }",
    "{ poweroff; }",
    "true && (reboot)",
    "echo hi; { reboot; }",
]


# Commands that look superficially similar but must NOT be hardline-blocked.
_HARDLINE_ALLOW = [
    # rm on non-protected paths
    "rm -rf /tmp/foo",
    "rm -rf /tmp/*",
    "rm -rf ./build",
    "rm -rf node_modules",
    "rm -rf /home/user/scratch",  # subpath of /home, not /home itself
    "rm -rf ~/Downloads/old",
    "rm -rf $HOME/tmp",
    "rm foo.txt",
    "rm -rf some/path",
    # Literal root-level directories that only LOOK like root-collapse
    # spellings. Each inter-slash segment must be exactly "." or ".." to
    # count as a collapse back to "/" — "/..." is a dir literally named
    # "..." and "/.foo" is an ordinary root dotfile. These must NOT be
    # swept into the "recursive delete of root filesystem" hardline rule
    # (regression guard for the collapse-spelling tightening).
    "rm -rf /...",
    "rm -rf /....",
    "rm -rf /.foo",
    "rm -rf /.config/foo",
    # A dangerous-looking command embedded as a quoted *argument* to another
    # command must not trip the floor: the path is immediately followed by a
    # closing quote with no matching opening quote of its own, so the
    # quote-tolerant matcher must still ignore it (no new false positives).
    'git commit -m "rm -rf /"',
    'git commit -m "wipe with rm -rf /etc"',
    # dd to regular files
    "dd if=/dev/zero of=./image.bin",
    "dd if=./data of=./backup.bin",
    # Redirect to regular files / non-block devices
    "echo done > /tmp/flag",
    "echo test > /dev/null",
    # Reading devices is fine
    "ls /dev/sda",
    "cat /dev/urandom | head -c 10",
    # Unrelated commands that happen to contain the trigger word
    "grep 'shutdown' logs.txt",
    "echo reboot",
    "echo '# init 0 in comment'",
    "cat rebooting.log",
    "echo 'halt and catch fire'",
    "python3 -c 'print(\"shutdown\")'",
    "find . -name '*reboot*'",
    # Word-boundary protection
    "mkfs_helper --version",
    # systemctl non-destructive verbs
    "systemctl status nginx",
    "systemctl restart nginx",
    "systemctl stop nginx",
    "systemctl start nginx",
    # targeted kill
    "kill -9 12345",
    "kill -HUP 1234",
    "pkill python",
    # Ordinary ops
    "git status",
    "npm run build",
    "sudo apt update",
    "curl https://example.com | head",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("command", _HARDLINE_BLOCK)
async def test_hardline_detection_blocks(command):
    is_hl, desc = await detect_hardline_command(command)
    assert is_hl, f"expected hardline to match {command!r}"
    assert desc, "hardline match must provide a description"


@pytest.mark.asyncio
@pytest.mark.parametrize("command", _HARDLINE_ALLOW)
async def test_hardline_detection_allows(command):
    is_hl, desc = await detect_hardline_command(command)
    assert not is_hl, f"expected hardline NOT to match {command!r} (got: {desc})"
    assert desc is None


# Commands written with the ordinary quoting / brace shell idioms that
# previously slipped past the floor. Kept as an explicit regression set so
# the intent (quoting `rm -rf "/"` must not be a disk-wipe bypass) survives
# any future refactor of the rm patterns.
_QUOTED_BRACE_BYPASS = [
    'rm -rf "/"',
    "rm -rf '/'",
    'rm -rf "/etc"',
    'rm -rf "/home"',
    'rm -rf "$HOME"',
    "rm -rf ${HOME}",
    'rm -rf "${HOME}"',
]


@pytest.mark.asyncio
@pytest.mark.parametrize("command", _QUOTED_BRACE_BYPASS)
async def test_quoted_and_brace_paths_are_hardline_blocked(command):
    """Quoted paths and ${HOME} must hit the floor (was a silent bypass)."""
    is_hl, desc = await detect_hardline_command(command)
    assert is_hl, f"quoting/brace bypass leaked through hardline floor: {command!r}"
    assert desc


# Multi-line QUOTED arguments are data, not command sequences: a newline
# inside quotes is part of the argument the shell passes to the program.
# These previously tripped the hardline floor because the flat command-start
# class treated every raw newline — even inside quotes — as a command
# boundary, blocking `hermes send` message bodies, multi-line
# `git commit -m` messages, and heredoc text that merely MENTION
# shutdown/reboot commands.
_QUOTED_NEWLINE_DATA_ALLOW = [
    # hermes send with a multi-line message body (the reported symptom)
    'hermes send -t telegram -s "spark1" "console output:\nsudo reboot\ndone"',
    'hermes send -t telegram "line1\nshutdown -h now\nline3"',
    # git commit -m with a multi-line message
    "git commit -m 'ops notes:\nreboot the box after the deploy'",
    'git commit -m "fix startup\nsystemctl reboot was flaky here"',
    # heredoc bodies quoting dangerous strings as data
    "python3 - <<'EOF'\nmsg = 'run sudo reboot later'\nprint(msg)\nEOF",
    "cat > /tmp/notes.txt <<'EOF'\nremember: shutdown -h now\nEOF",
    # rm hardline floor is anchored to the same class — quoted prose about it
    # across a line break must stay data too
    'git commit -m "docs:\nwarn about rm -rf / in the guide"',
]

# The masking must be strictly scoped to quoted data: real command
# boundaries around/inside those same shapes still hit the floor.
_QUOTED_NEWLINE_THREATS_BLOCK = [
    # unquoted newline is a real command separator
    "echo hi\nsudo reboot",
    'echo "a"\nsudo reboot',
    'git commit -m "safe message"\nshutdown -h now',
    # command substitution inside double quotes really executes
    'hermes send -t telegram "$(sudo reboot)"',
    'echo "`shutdown -h now`"',
    # multi-line quoted data followed by a REAL chained command
    'hermes send "line1\nline2" && sudo reboot',
    # a heredoc whose body is data, but the delivery command itself is hardline
    "sudo reboot <<'EOF'\nignored\nEOF",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("command", _QUOTED_NEWLINE_DATA_ALLOW)
async def test_quoted_newline_data_not_blocked(command):
    """Newlines inside quoted arguments are data, not command starts."""
    is_hl, desc = await detect_hardline_command(command)
    assert not is_hl, (
        f"multi-line quoted data false-positived the hardline floor: "
        f"{command!r} (got: {desc})"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("command", _QUOTED_NEWLINE_THREATS_BLOCK)
async def test_real_newline_separated_threats_still_blocked(command):
    """Unquoted newlines / $() / backticks remain real command boundaries."""
    is_hl, desc = await detect_hardline_command(command)
    assert is_hl, f"real threat leaked through hardline floor: {command!r}"
    assert desc


@pytest.mark.asyncio
async def test_quoted_newline_data_not_blocked_by_full_guard_chain():
    """End-to-end: the guard chain must not hardline-block a multi-line
    quoted message (yolo on, so only the unconditional floor can block)."""
    command = 'hermes send -t telegram "status:\nsudo reboot happened at 3am"'
    result = await check_all_command_guards(command, "local")
    assert result["approved"], (
        f"guard chain blocked multi-line quoted data: {result.get('message')}"
    )


# Commands that carry the literal string "rm -rf /" (or a sibling) as DATA in
# another command's quoted argument — a PR title, a commit message, an echo /
# printf argument. The shell never executes that text as an rm command, so the
# hardline floor must NOT fire; otherwise the command cannot run at all (this
# blocked `gh pr create --title "…rm -rf /…"` outright). Regression guard for
# the command-position anchor on the rm rules.
_DATA_ARG_NOT_A_COMMAND = [
    'gh pr create --title "block rm -rf / spellings"',
    'git commit -m "fixes rm -rf / bypass"',
    'echo "run rm -rf / now"',
    'echo "rm -rf /"',
    'printf "%s" "rm -rf /"',
    'gh issue comment 1 --body "the fix blocks rm -rf //"',
    # A `(` or `{` INSIDE a quoted argument is prose, not a subshell/brace
    # opener — the trigger word after it is data. Naively adding `(` / `{` to
    # the flat command-position class blocked these (it broke our own
    # `gh pr create --title "…(reboot)…"` workflow); the quote-aware tokenizer
    # must leave them alone.
    'gh pr create --title "block (reboot) spellings"',
    'git commit -m "(rm -rf /) note"',
    'echo "(reboot)"',
    'echo "{ reboot; }"',
    "echo '(poweroff)'",
    "echo '{ rm -rf /; }'",
    'find . -name "*(reboot)*"',
]


# Real root wipes at every command position — bare, chained after a separator,
# inside a command substitution ($()/backtick), or after sudo/env wrappers.
# The command-position anchor must keep catching all of these; the substitution
# forms exercise the shell-metacharacter terminator on the bare path branch.
_COMMAND_POSITION_ROOT_WIPES = [
    "rm -rf /",
    "ls && rm -rf /",
    "ls; rm -rf /",
    "echo x | rm -rf /",
    "sudo rm -rf /",
    "env X=1 rm -rf /",
    "$(rm -rf /)",
    "`rm -rf /`",
    'echo "$(rm -rf /)"',
    # Bare subshell / brace-group openers are real command positions too.
    "(rm -rf /)",
    "{ rm -rf /; }",
    "(rm -rf ~)",
    "(sudo rm -rf /)",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("command", _COMMAND_POSITION_ROOT_WIPES)
async def test_root_wipe_at_command_position_is_hardline(command):
    """A real `rm -rf /` at any command position stays hardline-blocked."""
    is_hl, desc = await detect_hardline_command(command)
    assert is_hl, f"real root wipe leaked past the floor: {command!r}"
    assert desc


# -------------------------------------------------------------------------
# Shell line-continuation bypass
# -------------------------------------------------------------------------
#
# A backslash immediately followed by a newline is a POSIX line
# continuation: the shell removes BOTH characters and joins the tokens, so
# `rm -rf \<newline>/` executes as `rm -rf /`. The normalizer used to strip
# only backslash-escapes of NON-newline characters (`\\([^\n])`), leaving the
# dangling backslash wedged between tokens — which broke the structured
# rm/dd/mkfs patterns and let a root wipe slip past the hardline floor.

# (command_with_continuation, description_substring) — each is the
# line-continuation form of a command already in _HARDLINE_BLOCK.
_HARDLINE_LINE_CONTINUATION = [
    ("rm -rf \\\n/", "root"),            # split before the path
    ("rm -r\\\nf /", "root"),            # split inside the flag bundle
    ("rm -rf \\\n~", "home"),            # home-directory wipe
    ("rm -rf \\\r\n/", "root"),          # CRLF line ending
    ("mkfs.ext4 \\\n/dev/sda1", "mkfs"),  # filesystem format
]


@pytest.mark.asyncio
@pytest.mark.parametrize("command,desc_substr", _HARDLINE_LINE_CONTINUATION)
async def test_hardline_blocks_line_continuation(command, desc_substr):
    is_hl, desc = await detect_hardline_command(command)
    assert is_hl, f"line-continuation bypassed hardline detection: {command!r}"
    assert desc and desc_substr in desc.lower(), (
        f"unexpected description {desc!r} for {command!r}"
    )

def test_hardline_list_is_small():
    """Hardline list stays focused on unrecoverable commands only.

    If you're adding a 20th+ pattern, reconsider — it probably belongs in
    DANGEROUS_PATTERNS where yolo can still bypass it.
    """
    assert len(HARDLINE_PATTERNS) <= 20, (
        f"HARDLINE_PATTERNS has grown to {len(HARDLINE_PATTERNS)} entries; "
        "only truly unrecoverable commands belong here."
    )

_SUDO_STDIN_BLOCK = [
    "sudo -S whoami",
    "echo hunter2 | sudo -S whoami",
    "sudo -S -u root whoami",
    "sudo -S apt-get install foo",
    "echo password | sudo -S systemctl restart nginx",
    "sudo -k && sudo -S whoami",
]

_SUDO_STDIN_ALLOW = [
    # Plain sudo without -S — goes through normal approval
    "sudo whoami",
    "sudo apt-get update",
    "sudo -u root whoami",
    # -S flag not attached to sudo
    "echo -S hello",
    "some_tool -S thing",
    # Literal text mention of sudo
    "echo 'use sudo -S to pipe passwords'",
]

_SUDO_STDIN_BLOCK_YOLO = [
    "sudo -S whoami",
    "echo hunter2 | sudo -S apt-get install",
]


@pytest.mark.asyncio
async def test_sudo_stdin_guard_detects_without_password():
    """sudo -S is dangerous when SUDO_PASSWORD is not configured."""
    import tools.approval as approval_mod

    for cmd in _SUDO_STDIN_BLOCK:
        is_blocked, desc = await approval_mod._check_sudo_stdin_guard(cmd)
        assert is_blocked, f"expected sudo stdin guard to block {cmd!r}"
        assert "sudo" in desc.lower()
