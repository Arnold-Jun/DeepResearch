from __future__ import annotations

import time
from typing import Any
from src.models import model_manager
from src.orchestrator.aggregator import aggregate_round
from src.orchestrator.planner import PlanningLLM
from src.orchestrator.runner import RoundRunner
from src.orchestrator.scheduler import SchedulerLLM
from src.orchestrator.summarizer import SummarizerLLM
from src.orchestrator.state import OrchestratorConfig, OrchestratorState, TodoItem
from src.orchestrator.todo_ops import apply_todo_delta_parent_only, insert_subtasks_after_parent

from src.agent.agent_builder import build_agent


class TopLevelOrchestrator:
    """
    Top-level orchestrator:
    planning -> parent pick -> scheduler -> run_round (N parallel subtasks, single agent type) -> aggregate -> planning ...
    """

    def __init__(self, *, config, orchestrator_config: OrchestratorConfig | None = None):
        self.config = config
        self.ocfg = orchestrator_config or self._load_orchestrator_config()

        self.planning_model = model_manager.get_model(self.ocfg.planning_model_id)
        self.scheduler_model = model_manager.get_model(self.ocfg.scheduler_model_id)
        self.planner = PlanningLLM(model=self.planning_model)
        self.scheduler = SchedulerLLM(model=self.scheduler_model, max_parallelism=self.ocfg.max_parallelism)
        self.summarizer_model = model_manager.get_model(self.ocfg.summarizer_model_id)
        self.summarizer = SummarizerLLM(model=self.summarizer_model)

        self.runner = RoundRunner(
            agent_factory=self._agent_factory,
            max_parallelism=self.ocfg.max_parallelism,
            timeout_s=self.ocfg.subtask_timeout_seconds,
            output_max_chars=self.ocfg.subtask_output_max_chars,
        )

    def _load_orchestrator_config(self) -> OrchestratorConfig:
        cfg = self.config.get("orchestrator_config", {}) if hasattr(self.config, "get") else {}
        return OrchestratorConfig(
            max_rounds=int(cfg.get("max_rounds", 12)),
            deadline_seconds=int(cfg.get("deadline_seconds", 600)),
            max_parallelism=int(cfg.get("max_parallelism", 5)),
            max_failures_per_parent=int(cfg.get("max_failures_per_parent", 6)),
            subtask_timeout_seconds=int(cfg.get("subtask_timeout_seconds", 240)),
            subtask_output_max_chars=int(cfg.get("subtask_output_max_chars", 1800)),
            subtask_failure_threshold=float(cfg.get("subtask_failure_threshold", 0.5)),
            planning_model_id=str(cfg.get("planning_model_id", "qwen3-32b")),
            scheduler_model_id=str(cfg.get("scheduler_model_id", "qwen3-32b")),
            summarizer_model_id=str(cfg.get("summarizer_model_id", cfg.get("planning_model_id", "qwen3-32b"))),
        )

    def _pick_parent(self, state: OrchestratorState) -> TodoItem | None:
        for item in state.todo_list:
            if not item.is_parent:
                continue
            if item.status not in ("todo", "in_progress", "blocked"):
                continue
            if item.failure_count >= self.ocfg.max_failures_per_parent:
                continue
            return item
        return None

    async def _build_sub_agent(self, agent_type: str):
        agent_config_name = f"{agent_type}_config"
        agent_cfg = self.config.get(agent_config_name, None)
        if agent_cfg is None:
            raise ValueError(f"Missing config for {agent_type}: expected key {agent_config_name}")
        mcp_tools = None
        if hasattr(self.config, "mcp_tools_config"):
            try:
                from src.mcp.mcpadapt import MCPAdapt, AsyncToolAdapter

                mcpadapt = MCPAdapt(self.config.mcp_tools_config, AsyncToolAdapter())
                mcp_tools = await mcpadapt.tools()
            except Exception:
                mcp_tools = None

        from src.registry import TOOL
        return await build_agent(
            self.config,
            agent_cfg,
            default_tools=TOOL,
            default_mcp_tools=mcp_tools,
        )

    async def _run_subtask(self, agent_type: str, task: str) -> str:
        agent = await self._build_sub_agent(agent_type)
        return await agent.run(task)

    def _make_sync_agent_proxy(self, agent_type: str):
        orchestrator = self

        class _Proxy:
            async def run(self, task: str):
                return await orchestrator._run_subtask(agent_type, task)

        return _Proxy()

    def _agent_factory(self, agent_type: str):
        return self._make_sync_agent_proxy(agent_type)

    async def run(self, task: str) -> str:
        state = OrchestratorState(
            task=task,
            todo_list=[],
            round_index=0,
            deadline_ts=time.time() + self.ocfg.deadline_seconds,
        )

        for r in range(self.ocfg.max_rounds):
            if state.is_timed_out():
                return "Reached global time limit."

            state.round_index = r + 1

            planning_delta = await self.planner.plan(state)
            apply_todo_delta_parent_only(state.todo_list, planning_delta)
            
            all_parents_done = all(
                item.status in ("done", "failed") 
                for item in state.todo_list 
                if item.is_parent
            )
            if all_parents_done and state.todo_list:
                from src.logger import logger
                logger.info("[Orchestrator] 所有父任务已完成，使用 Summarizer 生成最终答案")
                try:
                    return await self.summarizer.summarize(state)
                except Exception as e:
                    logger.error(f"[Orchestrator] Summarizer 失败: {e}", exc_info=True)
                    completed_subtasks = [item for item in state.todo_list if not item.is_parent and item.status == "done"]
                    if completed_subtasks:
                        return f"任务已完成。共完成 {len(completed_subtasks)} 个子任务。Summarizer 生成答案时出错，请查看子任务执行结果。"
                    return "任务已完成，但 Summarizer 生成答案时出错，且没有可用的子任务结果。"

            parent = self._pick_parent(state)
            if parent is None:
                state.last_round_summary = (
                    "No executable PARENT todo items found. "
                    "Please add parent tasks (op=add) derived from the user task. "
                    "If all tasks are already completed, the system will automatically call the summarizer."
                )
                continue

            # 3) Scheduler produces round plan for this parent
            plan = await self.scheduler.schedule(state, parent)
            if plan.get("route") == "finish":
                # Scheduler 判断可以完成，使用 Summarizer 生成最终答案
                from src.logger import logger
                logger.info("[Orchestrator] Scheduler 判断任务可完成，使用 Summarizer 生成最终答案")
                try:
                    return await self.summarizer.summarize(state)
                except Exception as e:
                    logger.error(f"[Orchestrator] Summarizer 失败: {e}", exc_info=True)
                    completed_subtasks = [item for item in state.todo_list if not item.is_parent and item.status == "done"]
                    if completed_subtasks:
                        return f"Scheduler 判断任务可完成。共完成 {len(completed_subtasks)} 个子任务。Summarizer 生成答案时出错，请查看子任务执行结果。"
                    return "Scheduler 判断任务可完成，但 Summarizer 生成答案时出错，且没有可用的子任务结果。"

            agent_type = str(plan.get("agent_type"))
            tasks = plan.get("tasks") or []
            if not isinstance(tasks, list):
                tasks = []
            tasks = tasks[:self.ocfg.max_parallelism]

            normalized_subtasks: list[TodoItem] = []
            
            existing_subtasks = [it for it in state.todo_list if it.parent_id == parent.id]
            existing_ids = {it.id for it in state.todo_list}
            start_index = len(existing_subtasks) + 1
            
            for idx, t in enumerate(tasks):
                if not isinstance(t, dict):
                    continue
                if str(t.get("parent_id")) != parent.id:
                    continue
                sub_task = str(t.get("task") or "").strip()
                if not sub_task:
                    continue
                
                sub_index = start_index + idx
                sub_id = f"{parent.id}_{sub_index}"
                
                if sub_id in existing_ids:
                    suffix = 1
                    base = sub_id
                    while f"{base}__{suffix}" in existing_ids:
                        suffix += 1
                    sub_id = f"{base}__{suffix}"
                
                existing_ids.add(sub_id)
                
                normalized_subtasks.append(
                    TodoItem(
                        id=sub_id,
                        task=sub_task,
                        status="todo",
                        parent_id=parent.id,
                        agent_type=agent_type,
                    )
                )

            if not normalized_subtasks:
                # treat as one failure on parent and continue
                parent.failure_count += 1
                parent.last_result_summary = "Scheduler produced no valid subtasks."
                parent.updated_at = time.time()
                state.last_round_summary = parent.last_result_summary
                continue

            insert_subtasks_after_parent(state.todo_list, parent.id, normalized_subtasks)

            results = await self.runner.run_round(agent_type, normalized_subtasks)

            failed = 0
            succeeded = 0
            id_to_item = {it.id: it for it in state.todo_list}
            for res in results:
                item = id_to_item.get(res.subtask_id)
                if not item:
                    continue
                item.updated_at = time.time()
                if res.ok:
                    item.status = "done"
                    item.last_result_summary = res.output
                    item.last_error = None
                    succeeded += 1
                else:
                    item.status = "failed"
                    item.last_error = res.error
                    item.last_result_summary = None
                    failed += 1

            total_subtasks = failed + succeeded
            if total_subtasks > 0:
                failure_rate = failed / total_subtasks
                success_rate = succeeded / total_subtasks
                
                from src.logger import logger
                logger.info(
                    f"父任务 {parent.id} 子任务执行结果：总数={total_subtasks}, "
                    f"成功={succeeded}, 失败={failed}, "
                    f"成功率={success_rate:.2%}, 失败率={failure_rate:.2%}"
                )
                
                # 判断父任务是否成功：成功率必须 >= (1 - 失败率阈值)
                required_success_rate = 1.0 - self.ocfg.subtask_failure_threshold
                if success_rate >= required_success_rate:
                    # 成功率满足要求，父任务可以标记为 done（由 Planner 判断）
                    logger.info(
                        f"父任务 {parent.id} 成功率 {success_rate:.2%} >= 要求 {required_success_rate:.2%}，"
                        f"可以标记为 done"
                    )
                else:
                    # 成功率不满足要求，增加失败计数
                    parent.failure_count += 1
                    logger.warning(
                        f"父任务 {parent.id} 成功率 {success_rate:.2%} < 要求 {required_success_rate:.2%}，"
                        f"failure_count 增加到 {parent.failure_count}"
                    )
            
            parent.updated_at = time.time()

            summary = aggregate_round(parent.id, agent_type, results)
            parent.last_result_summary = summary
            state.last_round_summary = summary
            state.last_round_agent_type = agent_type
            state.last_round_parent_id = parent.id

        from src.logger import logger
        logger.info("[Orchestrator] 达到最大轮次，使用 Summarizer 生成最终答案")
        try:
            return await self.summarizer.summarize(state)
        except Exception as e:
            logger.error(f"[Orchestrator] Summarizer 失败: {e}", exc_info=True)
            completed_subtasks = [item for item in state.todo_list if not item.is_parent and item.status == "done"]
            if completed_subtasks:
                return f"达到最大轮次。共完成 {len(completed_subtasks)} 个子任务。Summarizer 生成答案时出错，请查看子任务执行结果。"
            return "达到最大轮次，但 Summarizer 生成答案时出错，且没有可用的子任务结果。"


