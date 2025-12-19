from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, TypedDict, Union, Optional
from pydantic import BaseModel, ConfigDict

from src.models import ChatMessage, MessageRole
from src.exception import AgentError
from src.utils import make_json_serializable
from src.logger import LogLevel, AgentLogger, Timing, TokenUsage

# 如果可用，导入 PIL.Image 用于模型重建
try:
    import PIL.Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

if TYPE_CHECKING:
    import PIL.Image


class ToolCall(BaseModel):
    name: str
    arguments: Any
    id: str

    def model_dump(self, **kwargs):
        """自定义转储以匹配预期格式。"""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": make_json_serializable(self.arguments),
            },
        }


class MemoryStep(BaseModel):
    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        raise NotImplementedError


class ActionStep(MemoryStep):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    step_number: int
    timing: Timing
    model_input_messages: list[ChatMessage] | None = None
    tool_calls: list[ToolCall] | None = None
    error: AgentError | None = None
    model_output_message: ChatMessage | None = None
    model_output: str | None = None
    observations: str | None = None
    observations_images: list["PIL.Image.Image"] | None = None
    action_output: Any = None
    token_usage: TokenUsage | None = None
    is_final_answer: bool = False

    def model_dump(self, **kwargs):
        # 我们重写此方法以手动解析 tool_calls 和 action_output
        return {
            "step_number": self.step_number,
            "timing": self.timing.model_dump(),
            "model_input_messages": self.model_input_messages,
            "tool_calls": [tc.model_dump() for tc in self.tool_calls] if self.tool_calls else [],
            "error": self.error.dict() if self.error else None,  # AgentError 有自定义的 dict() 方法
            "model_output_message": self.model_output_message.model_dump() if self.model_output_message else None,
            "model_output": self.model_output,
            "observations": self.observations,
            "observations_images": [image.tobytes() for image in self.observations_images]
            if self.observations_images
            else None,
            "action_output": make_json_serializable(self.action_output),
            "token_usage": self.token_usage.model_dump() if self.token_usage else None,
            "is_final_answer": self.is_final_answer,
        }

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        messages = []
        if self.model_output is not None and not summary_mode:
            messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.model_output.strip()}])
            )

        if self.tool_calls is not None:
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL_CALL,
                    content=[
                        {
                            "type": "text",
                            "text": "调用工具：\n" + str([tc.model_dump() for tc in self.tool_calls]),
                        }
                    ],
                )
            )

        if self.observations_images:
            messages.append(
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "image",
                            "image": image,
                        }
                        for image in self.observations_images
                    ],
                )
            )

        if self.observations is not None:
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[
                        {
                            "type": "text",
                            "text": f"观察:\n{self.observations}",
                        }
                    ],
                )
            )
        if self.error is not None:
            error_message = (
                "错误：\n"
                + str(self.error)
                + "\n现在让我们重试：注意不要重复之前的错误！如果您已经重试了几次，请尝试完全不同的方法。\n"
            )
            message_content = f"Call id: {self.tool_calls[0].id}\n" if self.tool_calls else ""
            message_content += error_message
            messages.append(
                ChatMessage(role=MessageRole.TOOL_RESPONSE, content=[{"type": "text", "text": message_content}])
            )

        return messages

class PlanningStep(MemoryStep):
    model_input_messages: list[ChatMessage]
    model_output_message: ChatMessage
    plan: str
    timing: Timing
    token_usage: TokenUsage | None = None

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        if summary_mode:
            return []
        return [
            ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.plan.strip()}]),
            ChatMessage(
                role=MessageRole.USER, content=[{"type": "text", "text": "现在继续执行此计划。"}]
            ),
            # 第二条消息创建角色变更，以防止模型简单地继续计划消息
        ]

class TaskStep(MemoryStep):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    task: str
    task_images: list["PIL.Image.Image"] | None = None

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        content = [{"type": "text", "text": f"新任务:\n{self.task}"}]
        if self.task_images:
            content.extend([{"type": "image", "image": image} for image in self.task_images])

        return [ChatMessage(role=MessageRole.USER, content=content)]


class SystemPromptStep(MemoryStep):
    system_prompt: str

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        if summary_mode:
            return []
        return [ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}])]


class FinalAnswerStep(MemoryStep):
    output: Any
    
class UserPromptStep(MemoryStep):
    user_prompt: str

    def to_messages(self, summary_mode: bool = False, **kwargs) -> List[ChatMessage]:
        if summary_mode:
            return []
        return [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": self.user_prompt}])]


class AgentMemory:
    def __init__(self, system_prompt: str, user_prompt: Optional[str] = None):
        self.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        if user_prompt is not None:
            self.user_prompt = UserPromptStep(user_prompt=user_prompt)
        else:
            self.user_prompt = None
        self.steps: list[TaskStep | ActionStep | PlanningStep] = []

    def reset(self):
        self.steps = []

    def get_normal_steps(self) -> list[dict]:
        return [
            {key: value for key, value in step.model_dump().items() if key != "model_input_messages"} for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        return [step.model_dump() for step in self.steps]


# 重建模型以解析对 PIL.Image.Image 的前向引用
# 在使用字符串类型注解时，这对于 Pydantic 2.x 是必需的
if PIL_AVAILABLE:
    TaskStep.model_rebuild()
    ActionStep.model_rebuild()
else:
    # 即使 PIL 不可用，我们仍然可以使用 Any 类型重建
    # 模型将工作，但 task_images/observations_images 将为 None
    TaskStep.model_rebuild()
    ActionStep.model_rebuild()

__all__ = ["AgentMemory"]
