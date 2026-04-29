"""命令行入口 — Agent 路线已移除，重定向到 pipeline CLI"""

from __future__ import annotations

import sys

from rich.console import Console

console = Console()


def app() -> None:
    """CLI 主入口。Agent 路线已移除，提示使用 pipeline CLI。"""
    args = sys.argv[1:]

    if args and args[0] in ("-h", "--help"):
        console.print(
            "[bold]ship-hull[/bold] — 船弦号识别系统\n\n"
            "Agent 路线已移除，请使用 pipeline CLI：\n"
            "  python -m pipeline.cli <source> [options]\n\n"
            "示例：\n"
            "  python -m pipeline.cli video.mp4 --demo\n"
            "  python -m pipeline.cli 0 --concurrent --enable-locate\n"
        )
        return

    console.print("[yellow]Agent 路线已移除。[/yellow]")
    console.print("请使用 pipeline CLI：")
    console.print("  [cyan]python -m pipeline.cli <source> [options][/cyan]")
    console.print()
    console.print("示例：")
    console.print("  python -m pipeline.cli video.mp4 --demo")
    console.print("  python -m pipeline.cli 0 --concurrent --enable-locate")
