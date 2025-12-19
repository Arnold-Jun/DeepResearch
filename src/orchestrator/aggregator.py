from __future__ import annotations

from typing import Any

from src.orchestrator.runner import SubtaskRunResult


def aggregate_round(parent_id: str, agent_type: str, results: list[SubtaskRunResult]) -> str:
    ok = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]

    lines: list[str] = []
    lines.append(f"Round summary: parent={parent_id} agent_type={agent_type}")
    lines.append(f"- ok: {len(ok)}")
    lines.append(f"- failed: {len(bad)}")
    lines.append("")

    if ok:
        lines.append("Successful outputs (truncated):")
        for r in ok:
            lines.append(f"- subtask_id={r.subtask_id}")
            if r.output:
                lines.append(r.output)
            lines.append("")

    if bad:
        lines.append("Failures:")
        for r in bad:
            lines.append(f"- subtask_id={r.subtask_id}: {r.error}")

    return "\n".join(lines).strip()


