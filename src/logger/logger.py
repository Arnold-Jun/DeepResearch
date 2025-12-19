import logging
import json
from enum import IntEnum
from typing import List, Optional

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from src.utils import (
    escape_code_brackets,
    Singleton
)

YELLOW_HEX = "#d4b702"

class LogLevel(IntEnum):
    OFF = -1  # 无输出
    ERROR = 0  # 仅错误
    INFO = 1  # 正常输出（默认）
    DEBUG = 2  # 详细输出

class AgentLogger(logging.Logger, metaclass=Singleton):
    def __init__(self, name="logger", level=logging.INFO):
        # Initialize the parent class
        super().__init__(name, level)

        # Define a formatter for log messages
        self.formatter = logging.Formatter(
            fmt="\033[92m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s",
            datefmt="%H:%M:%S",
        )

    def init_logger(self, log_path: str, level=logging.INFO, use_stderr: bool = False):
        """
        使用文件路径和可选的主进程检查初始化日志记录器。

        参数:
            log_path (str): 日志文件路径。
            level (int, 可选): 日志级别。默认为 logging.INFO。
            use_stderr (bool, 可选): 如果为 True，输出到 stderr 而不是 stdout。
                                      对于使用 stdout 进行 JSON-RPC 的 MCP 服务器很有用。
            accelerator (Accelerator, 可选): 用于确定主进程的 Accelerator 实例。
        """
        import sys
        
        # Add a console handler for logging to the console
        # Use stderr if requested (e.g., for MCP servers that use stdout for JSON-RPC)
        stream = sys.stderr if use_stderr else sys.stdout
        console_handler = logging.StreamHandler(stream)
        console_handler.setLevel(level)
        console_handler.setFormatter(self.formatter)
        self.addHandler(console_handler)

        # 添加文件处理器以记录到文件
        file_handler = logging.FileHandler(
            log_path, mode="a", encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(self.formatter)
        self.addHandler(file_handler)

        self.console = Console(width=100)
        self.file_console = Console(file=open(log_path, "a", encoding="utf-8"), width=100)

        # Prevent duplicate logs from propagating to the root logger
        self.propagate = False

    def log(self, *args, level: int | str | LogLevel = LogLevel.INFO, **kwargs) -> None:
        """将消息记录到控制台。

        参数:
            level (LogLevel, 可选): 默认为 LogLevel.INFO。
        """
        if isinstance(level, str):
            level = LogLevel[level.upper()]
        if level <= self.level:
            self.info(*args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        Overridden info method with stacklevel adjustment for correct log location.
        """
        if isinstance(msg, (Rule, Panel, Group, Tree, Table, Syntax)):
            self.console.print(msg)
            self.file_console.print(msg)
        else:
            kwargs.setdefault(
                "stacklevel", 2
            )  # 调整堆栈级别以显示实际调用者
            if "style" in kwargs:
                kwargs.pop("style")
            if "level" in kwargs:
                kwargs.pop("level")
            super().info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        super().warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        super().error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        super().critical(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        kwargs.setdefault("stacklevel", 2)
        super().debug(msg, *args, **kwargs)

    def log_error(self, error_message: str) -> None:
        self.info(escape_code_brackets(error_message), style="bold red", level=LogLevel.ERROR)

    def log_markdown(self, content: str, title: str | None = None, level=LogLevel.INFO, style=YELLOW_HEX) -> None:
        markdown_content = Syntax(
            content,
            lexer="markdown",
            theme="github-dark",
            word_wrap=True,
        )
        if title:
            self.info(
                Group(
                    Rule(
                        "[bold italic]" + title,
                        align="left",
                        style=style,
                    ),
                    markdown_content,
                ),
                level=level,
            )
        else:
            self.info(markdown_content, level=level)

    def log_code(self, title: str, content: str, level: int = LogLevel.INFO) -> None:
        self.info(
            Panel(
                Syntax(
                    content,
                    lexer="python",
                    theme="monokai",
                    word_wrap=True,
                ),
                title="[bold]" + title,
                title_align="left",
                box=box.HORIZONTALS,
            ),
            level=level,
        )

    def log_rule(self, title: str, level: int = LogLevel.INFO) -> None:
        self.info(
            Rule(
                "[bold]" + title,
                characters="━",
                style=YELLOW_HEX,
            ),
            level=LogLevel.INFO,
        )

    def log_task(self, content: str, subtitle: str, title: str | None = None, level: LogLevel = LogLevel.INFO) -> None:
        self.info(
            Panel(
                f"\n[bold]{escape_code_brackets(content)}\n",
                title="[bold]New run" + (f" - {title}" if title else ""),
                subtitle=subtitle,
                border_style=YELLOW_HEX,
                subtitle_align="left",
            ),
            level=level,
        )

    def log_messages(self, messages: list[dict], level: LogLevel = LogLevel.DEBUG) -> None:
        messages_as_string = "\n".join([json.dumps(dict(message), indent=4, ensure_ascii=False) for message in messages])
        self.info(
            Syntax(
                messages_as_string,
                lexer="markdown",
                theme="github-dark",
                word_wrap=True,
            ),
            level=level,
        )

    def visualize_agent_tree(self, agent):
        def create_tools_section(tools_dict):
            table = Table(show_header=True, header_style="bold")
            table.add_column("Name", style="#1E90FF")
            table.add_column("Description")
            table.add_column("Arguments")

            for name, tool in tools_dict.items():
                args = [
                    f"{arg_name} (`{info.get('type', 'Any')}`{', optional' if info.get('optional') else ''}): {info.get('description', '')}"
                    for arg_name, info in getattr(tool, "inputs", {}).items()
                ]
                table.add_row(name, getattr(tool, "description", str(tool)), "\n".join(args))

            return Group("🛠️ [italic #1E90FF]Tools:[/italic #1E90FF]", table)

        def get_agent_headline(agent, name: str | None = None):
            name_headline = f"{name} | " if name else ""
            return f"[bold {YELLOW_HEX}]{name_headline}{agent.__class__.__name__} | {agent.model.model_id}"

        def build_agent_tree(parent_tree, agent_obj):
            """递归构建智能体树。"""
            parent_tree.add(create_tools_section(agent_obj.tools))

            if agent_obj.sub_agents:
                agents_branch = parent_tree.add("🤖 [italic #1E90FF]Sub agents:")
                for name, sub_agent in agent_obj.sub_agents.items():
                    agent_tree = agents_branch.add(get_agent_headline(sub_agent, name))
                    agent_tree.add(f"📝 [italic #1E90FF]Description:[/italic #1E90FF] {sub_agent.description}")
                    build_agent_tree(agent_tree, sub_agent)

        main_tree = Tree(get_agent_headline(agent))
        build_agent_tree(main_tree, agent)
        self.console.print(main_tree)

logger = AgentLogger()