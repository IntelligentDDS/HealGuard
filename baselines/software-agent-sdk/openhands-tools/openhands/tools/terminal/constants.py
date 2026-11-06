import os
import re
from typing import Final


CMD_OUTPUT_PS1_BEGIN: Final[str] = "\n###PS1JSON###\n"
CMD_OUTPUT_PS1_END: Final[str] = "\n###PS1END###"
# Regex to match PS1 metadata blocks. Uses negative lookahead to handle corruption
# scenarios where concurrent output causes nested ###PS1JSON### markers. This ensures
# we match only the LAST ###PS1JSON### before each ###PS1END###.
CMD_OUTPUT_METADATA_PS1_REGEX: Final[re.Pattern[str]] = re.compile(
    rf"^{CMD_OUTPUT_PS1_BEGIN.strip()}((?:(?!{CMD_OUTPUT_PS1_BEGIN.strip()}).)*?){CMD_OUTPUT_PS1_END.strip()}",
    re.DOTALL | re.MULTILINE,
)

# Default max size for command output content
# to prevent too large observations from being saved in the stream
# This matches the default max_message_chars in LLM class
MAX_CMD_OUTPUT_SIZE: Final[int] = 30000


# Common timeout message that can be used across different timeout scenarios
TIMEOUT_MESSAGE_TEMPLATE: Final[str] = (
    "You may wait longer to see additional output by sending empty command '', "
    "send other commands to interact with the current process, send keys "
    '("C-c", "C-z", "C-d") '
    "to interrupt/kill the previous command before sending your new command, "
    "or use the timeout parameter in terminal for future commands."
)

# How long to wait with no new output before considering it a no-change timeout
NO_CHANGE_TIMEOUT_SECONDS: Final[int] = 30

# How often to poll for new output in seconds
POLL_INTERVAL: Final[float] = 0.5
HISTORY_LIMIT: Final[int] = 10_000

# Per-PROCESS tmux socket. Must be unique per OS process: when several run_agents processes
# (e.g. one per model: claude / deepseek / gpt) drive OpenHands concurrently, a SHARED socket
# name made them all attach to ONE tmux server. Concurrent new-session / kill-session /
# list-sessions commands then race on that single server, libtmux sees TmuxObjectDoesNotExist /
# LibTmuxException, the pool's recovery path marks itself closed, and later checkouts raise
# "TmuxPanePool is not initialized or already closed" → the crash is force-skipped (corrupting
# results). Appending the PID gives each process its OWN tmux server (full isolation) while all
# pools/terminals WITHIN one process still share it (correct — that's intra-process pooling).
# Evaluated once at import time, so it is stable for the life of the process.
TMUX_SOCKET_NAME: Final[str] = f"openhands-{os.getpid()}"

# Tmux session dimensions (columns x rows). Keep the viewport wide enough for
# common command output while leaving scrollback retention to HISTORY_LIMIT.
# Output wider than TMUX_SESSION_WIDTH columns will wrap; this is an accepted
# tradeoff to avoid the oversized 1000x1000 virtual terminal grid.
TMUX_SESSION_WIDTH: Final[int] = 256
TMUX_SESSION_HEIGHT: Final[int] = 200
