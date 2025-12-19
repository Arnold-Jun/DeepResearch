from typing import Any
from collections.abc import Generator

from src.models.base import (ApiModel,
                             ChatMessage,
                             tool_role_conversions,
                             MessageRole,
                             TokenUsage,
                             ChatMessageStreamDelta,
                             ChatMessageToolCallStreamDelta)
from src.models.message_manager import MessageManager

class OpenAIServerModel(ApiModel):
    """此模型连接到 OpenAI 兼容的 API 服务器。

    参数:
        model_id (`str`):
            在服务器上使用的模型标识符（例如 "gpt-3.5-turbo"）。
        api_base (`str`, *可选*):
            OpenAI 兼容 API 服务器的基础 URL。
        api_key (`str`, *可选*):
            用于身份验证的 API 密钥。
        organization (`str`, *可选*):
            用于 API 请求的组织。
        project (`str`, *可选*):
            用于 API 请求的项目。
        client_kwargs (`dict[str, Any]`, *可选*):
            传递给 OpenAI 客户端的附加关键字参数（如 organization, project, max_retries 等）。
        custom_role_conversions (`dict[str, str]`, *可选*):
            自定义角色转换映射，用于转换其他消息角色。
            对于不支持特定消息角色（如 "system"）的特定模型很有用。
        flatten_messages_as_text (`bool`, 默认 `False`):
            是否将消息展平为文本。
        **kwargs:
            传递给 OpenAI API 的附加关键字参数。
    """

    def __init__(
        self,
        model_id: str,
        api_base: str | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool = False,
        http_client: Any = None,
        **kwargs,
        ):
        self.model_id = model_id
        self.api_base = api_base
        self.api_key = api_key
        flatten_messages_as_text = (
            flatten_messages_as_text
            if flatten_messages_as_text is not None
            else model_id.startswith(("ollama", "groq", "cerebras"))
        )

        self.http_client = http_client

        self.client_kwargs = {
            **(client_kwargs or {}),
            "api_key": api_key,
            "base_url": api_base,
            "organization": organization,
            "project": project,
        }

        self.message_manager = MessageManager(model_id=model_id)

        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):

        if self.http_client:
            return self.http_client
        else:
            try:
                import openai
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "请安装 'openai' 额外依赖以使用 OpenAIServerModel：`pip install 'smolagents[openai]'`"
                ) from e

            return openai.OpenAI(
                **self.client_kwargs
            )

    def _prepare_completion_kwargs(
            self,
            messages: list[ChatMessage],
            stop_sequences: list[str] | None = None,
            response_format: dict[str, str] | None = None,
            tools_to_call_from: list[Any] | None = None,
            custom_role_conversions: dict[str, str] | None = None,
            convert_images_to_image_urls: bool = False,
            tool_choice: str | dict | None = "required",  # 可配置的 tool_choice 参数
            **kwargs,
    ) -> dict[str, Any]:
        """
        准备模型调用所需的参数，处理参数优先级。

        参数优先级从高到低：
        1. 显式传递的 kwargs
        2. 特定参数（stop_sequences, response_format 等）
        3. self.kwargs 中的默认值
        """
        # 清理和标准化消息列表
        flatten_messages_as_text = kwargs.pop("flatten_messages_as_text", self.flatten_messages_as_text)
        messages_as_dicts = self.message_manager.get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=flatten_messages_as_text,
        )
        # 使用 self.kwargs 作为基础配置
        completion_kwargs = {
            **self.kwargs,
            "messages": messages_as_dicts,
        }

        # 处理特定参数
        if stop_sequences is not None:
            completion_kwargs["stop"] = stop_sequences
        if response_format is not None:
            completion_kwargs["response_format"] = response_format

        # 处理工具参数
        if tools_to_call_from:
            tools_config = {
                "tools": [self.message_manager.get_tool_json_schema(tool, model_id=self.model_id) for tool in
                          tools_to_call_from],
            }
            if tool_choice is not None:
                tools_config["tool_choice"] = tool_choice
            completion_kwargs.update(tools_config)

        # 最后，使用传入的 kwargs 覆盖所有设置
        completion_kwargs.update(kwargs)

        completion_kwargs = self.message_manager.get_clean_completion_kwargs(completion_kwargs)

        return completion_kwargs

    def generate_stream(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            http_client=self.http_client,
            **kwargs,
        )
        for event in self.client.chat.completions.create(
            **completion_kwargs, stream=True, stream_options={"include_usage": True}
        ):
            if event.usage:
                self._last_input_token_count = event.usage.prompt_tokens
                self._last_output_token_count = event.usage.completion_tokens
                yield ChatMessageStreamDelta(
                    content="",
                    token_usage=TokenUsage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    ),
                )
            if event.choices:
                choice = event.choices[0]
                if choice.delta:
                    yield ChatMessageStreamDelta(
                        content=choice.delta.content,
                        tool_calls=[
                            ChatMessageToolCallStreamDelta(
                                index=delta.index,
                                id=delta.id,
                                type=delta.type,
                                function=delta.function,
                            )
                            for delta in choice.delta.tool_calls
                        ]
                        if choice.delta.tool_calls
                        else None,
                    )
                else:
                    if not getattr(choice, "finish_reason", None):
                        raise ValueError(f"事件中没有内容或工具调用: {event}")

    async def generate(
            self,
            messages: list[ChatMessage],
            stop_sequences: list[str] | None = None,
            response_format: dict[str, str] | None = None,
            tools_to_call_from: list[Any] | None = None,
            **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )

        # 对于非流式调用，enable_thinking 必须设置为 false
        # 在 extra_body 中或直接在 completion_kwargs 中处理它
        if "extra_body" in completion_kwargs:
            if completion_kwargs["extra_body"] is None:
                completion_kwargs["extra_body"] = {}
            completion_kwargs["extra_body"]["enable_thinking"] = False
        elif "enable_thinking" in completion_kwargs:
            completion_kwargs["enable_thinking"] = False
        else:
            # 确保 extra_body 存在且 enable_thinking=False
            completion_kwargs.setdefault("extra_body", {})["enable_thinking"] = False

        response = await self.client.chat.completions.create(**completion_kwargs)

        self._last_input_token_count = response.usage.prompt_tokens
        self._last_output_token_count = response.usage.completion_tokens
        return ChatMessage.from_dict(
            response.choices[0].message.model_dump(include={"role", "content", "tool_calls"}),
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )

    async def __call__(self, *args, **kwargs) -> ChatMessage:
        """
        使用给定参数调用模型。
        这是一个便捷方法，使用相同的参数调用 `generate`。
        """
        return await self.generate(*args, **kwargs)