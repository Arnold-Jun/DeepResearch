# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import json5
import logging
import os
import re
import uuid
import warnings
from collections.abc import Generator
from copy import deepcopy
from pydantic import BaseModel, ConfigDict
from enum import Enum
from threading import Thread
from typing import TYPE_CHECKING, Any

from src.logger import TokenUsage
from src.utils import (_is_package_available,
                       encode_image_base64, 
                       make_image_url, 
                       parse_json_blob)


if TYPE_CHECKING:
    from transformers import StoppingCriteriaList


logger = logging.getLogger(__name__)

STRUCTURED_GENERATION_PROVIDERS = ["cerebras", "fireworks-ai"]


def get_dict_from_nested_pydantic(obj, ignore_key=None):
    """辅助函数：将嵌套的 pydantic 模型转换为字典，可选择排除键。"""
    def convert(obj):
        if isinstance(obj, BaseModel):
            data = obj.model_dump()
            if ignore_key and ignore_key in data:
                del data[ignore_key]
            return {k: convert(v) for k, v in data.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    return convert(obj)


class ChatMessageToolCallFunction(BaseModel):
    arguments: Any
    name: str
    description: str | None = None


class ChatMessageToolCall(BaseModel):
    function: ChatMessageToolCallFunction
    id: str
    type: str

    def __str__(self) -> str:
        return f"Call: {self.id}: Calling {str(self.function.name)} with arguments: {str(self.function.arguments)}"


class ChatMessage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    role: str
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[ChatMessageToolCall] | None = None
    raw: Any | None = None
    token_usage: TokenUsage | None = None

    def model_dump_json(self):
        return json.dumps(get_dict_from_nested_pydantic(self, ignore_key="raw"))

    @classmethod
    def model_validate(cls, obj: dict, raw: Any | None = None, token_usage: TokenUsage | None = None) -> "ChatMessage":
        """从字典创建 ChatMessage，可选择包含 raw 和 token_usage。"""
        if isinstance(obj, dict):
            data = obj.copy()
        else:
            data = obj
        
        if data.get("tool_calls"):
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(**tc["function"]), id=tc["id"], type=tc["type"]
                )
                for tc in data["tool_calls"]
            ]
            data["tool_calls"] = tool_calls
        
        if raw is not None:
            data["raw"] = raw
        if token_usage is not None:
            data["token_usage"] = token_usage
            
        return super().model_validate(data)

    @classmethod
    def from_dict(cls, data: dict, raw: Any | None = None, token_usage: TokenUsage | None = None) -> "ChatMessage":
        """从字典创建 ChatMessage，可选择包含 raw 和 token_usage。"""
        return cls.model_validate(data, raw=raw, token_usage=token_usage)

    def render_as_markdown(self) -> str:
        rendered = str(self.content) or ""
        if self.tool_calls:
            rendered += "\n".join(
                [
                    json.dumps({"tool": tool.function.name, "arguments": tool.function.arguments})
                    for tool in self.tool_calls
                ]
            )
        return rendered


def parse_json_if_needed(arguments: str | dict) -> str | dict:
    if isinstance(arguments, dict):
        return arguments
    else:
        try:
            return json5.loads(arguments)
        except Exception:
            return arguments


class ChatMessageToolCallStreamDelta(BaseModel):
    """表示生成期间工具调用的流式增量。"""

    index: int | None = None
    id: str | None = None
    type: str | None = None
    function: ChatMessageToolCallFunction | None = None


class ChatMessageStreamDelta(BaseModel):
    content: str | None = None
    tool_calls: list[ChatMessageToolCallStreamDelta] | None = None
    token_usage: TokenUsage | None = None


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool-call"
    TOOL_RESPONSE = "tool-response"

    @classmethod
    def roles(cls):
        return [r.value for r in cls]


def agglomerate_stream_deltas(
    stream_deltas: list[ChatMessageStreamDelta], role: MessageRole = MessageRole.ASSISTANT
) -> ChatMessage:
    """
    将流式增量列表聚合成单个聊天消息。
    """
    accumulated_tool_calls: dict[int, ChatMessageToolCallStreamDelta] = {}
    accumulated_content = ""
    total_input_tokens = 0
    total_output_tokens = 0
    for stream_delta in stream_deltas:
        if stream_delta.token_usage:
            total_input_tokens += stream_delta.token_usage.input_tokens
            total_output_tokens += stream_delta.token_usage.output_tokens
        if stream_delta.content:
            accumulated_content += stream_delta.content
        if stream_delta.tool_calls:
            for tool_call_delta in stream_delta.tool_calls:
                if tool_call_delta.index is not None:
                    if tool_call_delta.index not in accumulated_tool_calls:
                        accumulated_tool_calls[tool_call_delta.index] = ChatMessageToolCallStreamDelta(
                            id=tool_call_delta.id,
                            type=tool_call_delta.type,
                            function=ChatMessageToolCallFunction(name="", arguments=""),
                        )
                    tool_call = accumulated_tool_calls[tool_call_delta.index]
                    if tool_call_delta.id:
                        tool_call.id = tool_call_delta.id
                    if tool_call_delta.type:
                        tool_call.type = tool_call_delta.type
                    if tool_call_delta.function:
                        if tool_call_delta.function.name and len(tool_call_delta.function.name) > 0:
                            tool_call.function.name = tool_call_delta.function.name
                        if tool_call_delta.function.arguments:
                            tool_call.function.arguments += tool_call_delta.function.arguments
                else:
                    raise ValueError(f"工具增量中未提供调用索引: {tool_call_delta}")

    return ChatMessage(
        role=role,
        content=accumulated_content,
        tool_calls=[
            ChatMessageToolCall(
                function=ChatMessageToolCallFunction(
                    name=tool_call_stream_delta.function.name,
                    arguments=tool_call_stream_delta.function.arguments,
                ),
                id=tool_call_stream_delta.id or "",
                type="function",
            )
            for tool_call_stream_delta in accumulated_tool_calls.values()
            if tool_call_stream_delta.function
        ],
        token_usage=TokenUsage(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        ),
    )


tool_role_conversions = {
    MessageRole.TOOL_CALL: MessageRole.ASSISTANT,
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


def get_tool_json_schema(tool: Any) -> dict:
    properties = deepcopy(tool.parameters.get("properties", {}))
    required = []
    for key, value in properties.items():
        if value["type"] == "any":
            value["type"] = "string"
        if not ("nullable" in value and value["nullable"]):
            required.append(key)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def remove_stop_sequences(content: str, stop_sequences: list[str]) -> str:
    for stop_seq in stop_sequences:
        if content[-len(stop_seq) :] == stop_seq:
            content = content[: -len(stop_seq)]
    return content


def get_clean_message_list(
    message_list: list[ChatMessage],
    role_conversions: dict[MessageRole, MessageRole] | dict[str, str] = {},
    convert_images_to_image_urls: bool = False,
    flatten_messages_as_text: bool = False,
) -> list[dict[str, Any]]:
    """
    创建要作为 LLM 输入的消息列表。这些消息是字典，并且与 transformers LLM 聊天模板兼容。
    具有相同角色的后续消息将被连接为单个消息。

    参数:
        message_list (`list[dict[str, str]]`): 聊天消息列表。
        role_conversions (`dict[MessageRole, MessageRole]`, *可选*): 用于转换角色的映射。
        convert_images_to_image_urls (`bool`, 默认 `False`): 是否将图像转换为图像 URL。
        flatten_messages_as_text (`bool`, 默认 `False`): 是否将消息展平为文本。
    """
    output_message_list: list[dict[str, Any]] = []
    message_list = deepcopy(message_list)
    for message in message_list:
        role = message.role
        if role not in MessageRole.roles():
            raise ValueError(f"不正确的角色 {role}，目前仅支持 {MessageRole.roles()}。")

        if role in role_conversions:
            message.role = role_conversions[role]  # type: ignore
        if isinstance(message.content, list):
            for element in message.content:
                assert isinstance(element, dict), "错误：此元素应该是字典：" + str(element)
                if element["type"] == "image":
                    assert not flatten_messages_as_text, f"不能将图像与 {flatten_messages_as_text=} 一起使用"
                    if convert_images_to_image_urls:
                        element.update(
                            {
                                "type": "image_url",
                                "image_url": {"url": make_image_url(encode_image_base64(element.pop("image")))},
                            }
                        )
                    else:
                        element["image"] = encode_image_base64(element["image"])

        if len(output_message_list) > 0 and message.role == output_message_list[-1]["role"]:
            assert isinstance(message.content, list), "错误: 内容错误:" + str(message.content)
            if flatten_messages_as_text:
                output_message_list[-1]["content"] += "\n" + message.content[0]["text"]
            else:
                for el in message.content:
                    if el["type"] == "text" and output_message_list[-1]["content"][-1]["type"] == "text":
                        output_message_list[-1]["content"][-1]["text"] += "\n" + el["text"]
                    else:
                        output_message_list[-1]["content"].append(el)
        else:
            if flatten_messages_as_text:
                content = message.content[0]["text"]
            else:
                content = message.content
            output_message_list.append(
                {
                    "role": message.role,
                    "content": content,
                }
            )
    return output_message_list


def get_tool_call_from_text(text: str, tool_name_key: str, tool_arguments_key: str) -> ChatMessageToolCall:
    tool_call_dictionary, _ = parse_json_blob(text)
    try:
        tool_name = tool_call_dictionary[tool_name_key]
    except Exception as e:
        raise ValueError(
            f"在生成的工具调用中未找到键 {tool_name_key=}。得到的键：{list(tool_call_dictionary.keys())}"
        ) from e
    tool_arguments = tool_call_dictionary.get(tool_arguments_key, None)
    if isinstance(tool_arguments, str):
        tool_arguments = parse_json_if_needed(tool_arguments)
    return ChatMessageToolCall(
        id=str(uuid.uuid4()),
        type="function",
        function=ChatMessageToolCallFunction(name=tool_name, arguments=tool_arguments),
    )


def supports_stop_parameter(model_id: str) -> bool:
    """
    检查模型是否支持 `stop` 参数。

    推理模型 openai/o3 和 openai/o4-mini（及其版本变体）不支持。

    参数:
        model_id (`str`): 模型标识符（例如 "openai/o3", "o4-mini-2025-04-16"）

    返回:
        bool: 如果模型支持 stop 参数则为 True，否则为 False
    """
    model_name = model_id.split("/")[-1]
    pattern = r"^(o3[-\d]*|o4-mini[-\d]*)$"
    return not re.match(pattern, model_name)


class Model:
    def __init__(
        self,
        flatten_messages_as_text: bool = False,
        tool_name_key: str = "name",
        tool_arguments_key: str = "arguments",
        model_id: str | None = None,
        **kwargs,
    ):
        self.flatten_messages_as_text = flatten_messages_as_text
        self.tool_name_key = tool_name_key
        self.tool_arguments_key = tool_arguments_key
        self.kwargs = kwargs
        self._last_input_token_count: int | None = None
        self._last_output_token_count: int | None = None
        self.model_id: str | None = model_id

    @property
    def last_input_token_count(self) -> int | None:
        warnings.warn(
            "属性 last_input_token_count 已弃用，将在版本 1.20 中移除。 "
            "请改用 TokenUsage.input_tokens。",
            FutureWarning,
        )
        return self._last_input_token_count

    @property
    def last_output_token_count(self) -> int | None:
        warnings.warn(
            "属性 last_output_token_count 已弃用，将在版本 1.20 中移除。"
            "请改用 TokenUsage.output_tokens。",
            FutureWarning,
        )
        return self._last_output_token_count

    def _prepare_completion_kwargs(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        convert_images_to_image_urls: bool = False,
        tool_choice: str | dict | None = "required",
        **kwargs,
    ) -> dict[str, Any]:
        """
        准备模型调用所需的参数，处理参数优先级。

        参数优先级从高到低：
        1. 显式传递的 kwargs
        2. 特定参数（stop_sequences, response_format 等）
        3. self.kwargs 中的默认值
        """
        flatten_messages_as_text = kwargs.pop("flatten_messages_as_text", self.flatten_messages_as_text)
        messages_as_dicts = get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=flatten_messages_as_text,
        )
        completion_kwargs = {
            **self.kwargs,
            "messages": messages_as_dicts,
        }

        if stop_sequences is not None:
            if supports_stop_parameter(self.model_id or ""):
                completion_kwargs["stop"] = stop_sequences
        if response_format is not None:
            completion_kwargs["response_format"] = response_format

        if tools_to_call_from:
            tools_config = {
                "tools": [get_tool_json_schema(tool) for tool in tools_to_call_from],
            }
            if tool_choice is not None:
                tools_config["tool_choice"] = tool_choice
            completion_kwargs.update(tools_config)

        completion_kwargs.update(kwargs)

        return completion_kwargs

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """处理输入消息并返回模型的响应。

        参数:
            messages (`list[dict[str, str | list[dict]]] | list[ChatMessage]`):
                要处理的消息字典列表。每个字典应具有结构 `{"role": "user/system", "content": "message content"}`。
            stop_sequences (`List[str]`, *可选*):
                如果在模型输出中遇到，将停止生成的字符串列表。
            response_format (`dict[str, str]`, *可选*):
                在模型响应中使用的响应格式。
            tools_to_call_from (`List[Any]`, *可选*):
                模型可用于生成响应的工具列表。
            **kwargs:
                传递给底层模型的附加关键字参数。

        返回:
            `ChatMessage`: 包含模型响应的聊天消息对象。
        """
        raise NotImplementedError("此方法必须在子类中实现")

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def parse_tool_calls(self, message: ChatMessage) -> ChatMessage:
        """有时 API 不会将工具调用作为特定对象返回，因此我们需要解析它。"""
        message.role = MessageRole.ASSISTANT
        if not message.tool_calls:
            assert message.content is not None, "消息不包含内容和工具调用"
            message.tool_calls = [
                get_tool_call_from_text(message.content, self.tool_name_key, self.tool_arguments_key)
            ]
        assert len(message.tool_calls) > 0, "在模型输出中未找到工具调用"
        for tool_call in message.tool_calls:
            tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
        return message

    def to_dict(self) -> dict:
        """
        将模型转换为 JSON 兼容的字典。
        """
        model_dictionary = {
            **self.kwargs,
            "model_id": self.model_id,
        }
        for attribute in [
            "custom_role_conversion",
            "temperature",
            "max_tokens",
            "provider",
            "timeout",
            "api_base",
            "torch_dtype",
            "device_map",
            "organization",
            "project",
            "azure_endpoint",
        ]:
            if hasattr(self, attribute):
                model_dictionary[attribute] = getattr(self, attribute)

        dangerous_attributes = ["token", "api_key"]
        for attribute_name in dangerous_attributes:
            if hasattr(self, attribute_name):
                print(
                    f"出于安全原因，我们不会导出模型的 `{attribute_name}` 属性。请手动导出它。"
                )
        return model_dictionary

    @classmethod
    def from_dict(cls, model_dictionary: dict[str, Any]) -> "Model":
        return cls(**{k: v for k, v in model_dictionary.items()})


class ApiModel(Model):
    """
    基于 API 的语言模型基类。

    此类作为实现与外部 API 交互的模型的基础。它处理管理模型 ID、
    自定义角色映射和 API 客户端连接的通用功能。

    参数:
        model_id (`str`):
            要与 API 一起使用的模型标识符。
        custom_role_conversions (`dict[str, str]`, **可选**):
            在内部角色名称和 API 特定角色名称之间转换的映射。默认为 None。
        client (`Any`, **可选**):
            预配置的 API 客户端实例。如果未提供，将创建默认客户端。默认为 None。
        **kwargs: 传递给父类的附加关键字参数。
    """

    def __init__(
        self, model_id: str, custom_role_conversions: dict[str, str] | None = None, client: Any | None = None, **kwargs
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.custom_role_conversions = custom_role_conversions or {}
        self.client = client or self.create_client()

    def create_client(self):
        """为特定服务创建 API 客户端。"""
        raise NotImplementedError("子类必须实现此方法以创建客户端")

__all__ = [
    "MessageRole",
    "tool_role_conversions",
    "get_clean_message_list",
    "Model",
    "ApiModel",
    "ChatMessage",
]
