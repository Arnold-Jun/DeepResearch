from __future__ import annotations

from typing import Any
import time

from src.orchestrator.state import TodoItem, TodoStatus


ALLOWED_PARENT_PATCH_KEYS = {"task", "status", "last_result_summary"}


def _now() -> float:
    return time.time()


def find_by_id(todo_list: list[TodoItem], item_id: str) -> TodoItem | None:
    for item in todo_list:
        if item.id == item_id:
            return item
    return None


def apply_todo_delta_parent_only(todo_list: list[TodoItem], delta: dict[str, Any]) -> None:
    """
    Apply TodoDelta actions to PARENT tasks only.
    - No reordering (append-only for adds)
    - update/delete must target parent_id == None
    """
    actions = delta.get("actions") or []
    for action in actions:
        if not isinstance(action, dict):
            continue
        op = action.get("op")
        if op == "add":
            item = action.get("item") or {}
            item_id = str(item.get("id") or "").strip()
            task = str(item.get("task") or "").strip()
            status = item.get("status") or "todo"
            parent_id = item.get("parent_id")
            if parent_id is not None:
                # Planning must not add subtasks in V0
                continue
            if not item_id or not task:
                continue
            if find_by_id(todo_list, item_id):
                continue
            todo_list.append(
                TodoItem(
                    id=item_id,
                    task=task,
                    status=status,
                    parent_id=None,
                    failure_count=0,
                    last_result_summary=item.get("last_result_summary"),
                    created_at=_now(),
                    updated_at=_now(),
                )
            )
        elif op == "update":
            item_id = str(action.get("id") or "").strip()
            patch = action.get("patch") or {}
            if not item_id or not isinstance(patch, dict):
                continue
            target = find_by_id(todo_list, item_id)
            if not target or not target.is_parent:
                continue
            # Apply allowed keys only
            for k, v in patch.items():
                if k not in ALLOWED_PARENT_PATCH_KEYS:
                    continue
                if k == "status":
                    if v in ("todo", "in_progress", "done", "blocked", "failed"):
                        target.status = v  # type: ignore[assignment]
                elif k == "task":
                    target.task = str(v)
                elif k == "last_result_summary":
                    target.last_result_summary = str(v) if v is not None else None
            target.updated_at = _now()
        elif op == "delete":
            item_id = str(action.get("id") or "").strip()
            if not item_id:
                continue
            # only delete parent + its subtasks
            target = find_by_id(todo_list, item_id)
            if not target or not target.is_parent:
                continue
            todo_list[:] = [it for it in todo_list if it.id != item_id and it.parent_id != item_id]


def insert_subtasks_after_parent(todo_list: list[TodoItem], parent_id: str, subtasks: list[TodoItem]) -> None:
    """
    Insert subtasks right after the parent and its existing subtasks group.
    This does not reorder existing parents, only inserts new items.
    """
    # find parent index
    parent_idx = None
    for i, it in enumerate(todo_list):
        if it.id == parent_id and it.is_parent:
            parent_idx = i
            break
    if parent_idx is None:
        todo_list.extend(subtasks)
        return
    # find insertion point after contiguous subtasks of this parent
    insert_at = parent_idx + 1
    while insert_at < len(todo_list) and todo_list[insert_at].parent_id == parent_id:
        insert_at += 1
    todo_list[insert_at:insert_at] = subtasks


