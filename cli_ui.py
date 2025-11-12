"""Shared terminal UI helpers for the weather demo scripts."""

from __future__ import annotations

import json
import os
import shutil
import sys
import textwrap
from typing import Iterable

# ANSI style codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

PALETTE = {
    "accent": "\033[38;5;45m",
    "user": "\033[38;5;81m",
    "assistant": "\033[38;5;214m",
    "tool": "\033[38;5;141m",
    "info": "\033[38;5;110m",
    "warning": "\033[38;5;221m",
    "error": "\033[38;5;203m",
    "success": "\033[38;5;120m",
    "muted": "\033[38;5;247m",
}

ICONS = {
    "info": "[i]",
    "success": "[+]",
    "warning": "[!]",
    "error": "[x]",
}

DEFAULT_WIDTH = 90
MIN_WIDTH = 48


def _stdout_supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    stream = getattr(sys.stdout, "isatty", None)
    return bool(stream and stream())


USE_COLOR = _stdout_supports_color()


def _style_code(style: str) -> str:
    return PALETTE.get(style, "")


def color_text(
    text: str,
    style: str = "accent",
    *,
    bold: bool = False,
    dim: bool = False,
) -> str:
    if not USE_COLOR:
        return text
    codes: list[str] = []
    if bold:
        codes.append(BOLD)
    if dim:
        codes.append(DIM)
    color = _style_code(style)
    if color:
        codes.append(color)
    if not codes:
        return text
    return "".join(codes) + text + RESET


def prompt_label(role: str = "YOU", symbol: str = ">>") -> str:
    role_key = role.strip().lower()
    if role_key in {"you", "user", "human"}:
        style = "user"
    elif role_key in {"assistant", "ai", "bot"}:
        style = "assistant"
    else:
        style = "accent"
    label = f"{role.upper()} {symbol} "
    return color_text(label, style=style, bold=True)


def print_status(message: str, kind: str = "info") -> None:
    icon = ICONS.get(kind, "[*]")
    style = kind if kind in PALETTE else "info"
    text = f"{icon} {message}"
    print(color_text(text, style=style))


def _terminal_width() -> int:
    try:
        columns = shutil.get_terminal_size((DEFAULT_WIDTH, 20)).columns
    except OSError:
        columns = DEFAULT_WIDTH
    return max(MIN_WIDTH, min(columns, 100))


def print_banner(title: str, subtitle: str | None = None) -> None:
    width = _terminal_width()
    line = "=" * width
    print(color_text(line, style="accent"))
    print(color_text(title.center(width), style="accent", bold=True))
    if subtitle:
        print(color_text(subtitle.center(width), style="muted"))
    print(color_text(line, style="accent"))


def _stringify(body: str | Iterable[str] | None) -> str:
    if body is None:
        return ""
    if isinstance(body, str):
        return body
    if isinstance(body, (list, tuple, set)):
        return "\n".join(str(item) for item in body)
    if isinstance(body, dict):
        try:
            return json.dumps(body, indent=2, ensure_ascii=False)
        except TypeError:
            return str(body)
    return str(body)


def _wrap_lines(text: str, width: int) -> list[str]:
    if width <= 0:
        return [text]
    lines: list[str] = []
    for raw_line in text.splitlines() or [""]:
        stripped = raw_line.rstrip()
        if not stripped:
            lines.append("")
            continue
        wrapped = textwrap.wrap(
            stripped,
            width=width,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        lines.extend(wrapped or [""])
    return lines or [""]


def format_panel(
    title: str,
    body: str | Iterable[str] | None,
    *,
    style: str = "accent",
    width: int | None = None,
) -> str:
    max_width = width or _terminal_width()
    max_width = max(MIN_WIDTH, min(max_width, 100))
    inner_width = max_width - 4
    text = _stringify(body)
    content_lines = _wrap_lines(text, inner_width)
    full_width = inner_width + 4

    frame_color = style if style in PALETTE else "accent"
    top_border = "+" + "=" * (full_width - 2) + "+"
    bottom_border = "+" + "=" * (full_width - 2) + "+"
    header_text = (title.strip() or "MESSAGE")[: full_width - 4]
    header_line = header_text.center(full_width - 2)
    separator = "|" + "-" * (full_width - 2) + "|"

    lines = [
        color_text(top_border, style=frame_color),
        color_text("|", style=frame_color)
        + color_text(header_line, style="muted", bold=True)
        + color_text("|", style=frame_color),
        color_text(separator, style=frame_color),
    ]

    left = color_text("|", style=frame_color)
    right = color_text("|", style=frame_color)
    for line in content_lines:
        padded = line.ljust(inner_width)
        lines.append(f"{left} {padded} {right}")

    lines.append(color_text(bottom_border, style=frame_color))
    return "\n".join(lines)


def print_panel(
    title: str,
    body: str | Iterable[str] | None,
    *,
    style: str = "accent",
    width: int | None = None,
) -> None:
    print(format_panel(title, body, style=style, width=width))
