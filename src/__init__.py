"""
Top-level `src` package.

Important: avoid importing heavy/optional dependencies at import-time.
Tools are exposed via lazy imports so modules like `src.config` / `src.orchestrator`
can be imported even if some optional tool dependencies are missing.
"""

from importlib import import_module
from typing import Any


_EXPORTED = {
    "Tool",
    "ToolResult",
    "AsyncTool",
    "PythonInterpreterTool",
    "AutoBrowserUseTool",
    "make_tool_instance",
}


def __getattr__(name: str) -> Any:  # PEP 562
    if name in _EXPORTED:
        mod = import_module("src.tools")
        return getattr(mod, name)
    raise AttributeError(name)


__all__ = sorted(_EXPORTED)