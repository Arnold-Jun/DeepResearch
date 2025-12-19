#!/usr/bin/env python
# coding=utf-8

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
import ast
import inspect
import json
import logging
import os
import sys
import tempfile
import textwrap
import types
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from typing import Any, get_type_hints

from huggingface_hub import (
    CommitOperationAdd,
    create_commit,
    create_repo,
    get_collection,
    hf_hub_download,
    metadata_update,
)

from src.utils import (
    _convert_type_hints_to_json_schema,
    get_imports,
    get_json_schema,
    _is_package_available,
    get_source,
    instance_to_source,
    is_valid_name,
)

from src.tools.tool_validation import MethodChecker, validate_tool_attributes


if TYPE_CHECKING:
    import mcp


from src.logger import logger


def validate_after_init(cls):
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.validate_arguments()

    cls.__init__ = new_init
    return cls


AUTHORIZED_TYPES = [
    "string",
    "boolean",
    "integer",
    "number",
    "image",
    "audio",
    "array",
    "object",
    "any",
    "null",
]

CONVERSION_DICT = {"str": "string", "int": "integer", "float": "number"}

class ToolResult(BaseModel):
    """Represents the result of a tool execution."""

    output: Optional[Any] = None
    error: Optional[str] = None
    base64_image: Optional[str] = None
    system: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def __bool__(self):
        return any(getattr(self, field) for field in self.__fields__)

    def __add__(self, other: "ToolResult"):
        def combine_fields(
            field: Optional[str], other_field: Optional[str], concatenate: bool = True
        ):
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("无法合并工具结果")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )

    def __str__(self):
        return f"错误: {self.error}" if self.error else str(self.output)

    def __repr__(self):
        return self.__str__()

    def replace(self, **kwargs):
        """返回一个具有给定字段替换的新 ToolResult。"""
        # return self.copy(update=kwargs)
        return type(self)(**{**self.model_dump(), **kwargs})


class Tool:
    """
    智能体使用的函数基类。子类化此类并实现 `forward` 方法以及以下类属性：

    - **description** (`str`) -- 工具功能的简短描述，包括它期望的输入和将返回的输出。
      例如 '这是一个从 `url` 下载文件的工具。它以 `url` 作为输入，并返回文件中包含的文本'。
    - **name** (`str`) -- 将在智能体提示词中使用的工具名称。例如 `"text-classifier"` 或 `"web_searcher"`。
    - **inputs** (`Dict[str, Dict[str, Union[str, type, bool]]]`) -- 输入期望的模态字典。
      它有一个 `type` 键和一个 `description` 键。
      这由 `launch_gradio_demo` 使用或从您的工具创建空间，也可以用于生成工具的描述。
    - **output_type** (`type`) -- 工具输出的类型。这由 `launch_gradio_demo` 使用
      或从您的工具创建空间，也可以用于生成工具的描述。

    如果您的工具在使用前需要执行昂贵的操作（例如加载模型），您也可以重写方法 [`~Tool.setup`]。
    [`~Tool.setup`] 将在您第一次使用工具时调用，但不会在实例化时调用。
    """

    name: str
    description: str
    parameters: dict[str, dict[str, str | type | bool]]
    output_type: str

    def __init__(self, *args, **kwargs):
        self.is_initialized = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        validate_after_init(cls)

    def validate_arguments(self):
        required_attributes = {
            "description": str,
            "name": str,
            "parameters": dict,
            "output_type": str,
        }
        # Validate class attributes
        for attr, expected_type in required_attributes.items():
            attr_value = getattr(self, attr, None)
            if attr_value is None:
                raise TypeError(f"您必须设置属性 {attr}。")
            if not isinstance(attr_value, expected_type):
                raise TypeError(
                    f"属性 {attr} 应该具有类型 {expected_type.__name__}，但得到 {type(attr_value)}。"
                )
        # - 验证名称
        if not is_valid_name(self.name):
            raise Exception(
                f"无效的工具名称 '{self.name}'：必须是有效的 Python 标识符且不能是保留关键字"
            )

        assert "type" in self.parameters, "'parameters' 属性应该有一个 'type' 键。"
        assert "properties" in self.parameters, "'parameters' 属性应该有一个 'properties' 键。"

        properties = self.parameters["properties"]

        # 验证输入
        for input_name, input_content in properties.items():
            assert isinstance(input_content, dict), f"输入 '{input_name}' 应该是一个字典。"
            assert "type" in input_content and "description" in input_content, (
                f"输入 '{input_name}' 应该有 'type' 和 'description' 键，只有 {list(input_content.keys())}。"
            )
            if input_content["type"] not in AUTHORIZED_TYPES:
                raise Exception(
                    f"输入 '{input_name}': 类型 '{input_content['type']}' 不是授权值，应该是 {AUTHORIZED_TYPES} 之一。"
                )
        # Validate output type
        assert getattr(self, "output_type", None) in AUTHORIZED_TYPES

        # 验证 forward 函数签名，除了使用"通用"签名的工具（PipelineTool, SpaceToolWrapper, LangChainToolWrapper）
        if not (
            hasattr(self, "skip_forward_signature_validation")
            and getattr(self, "skip_forward_signature_validation") is True
        ):
            signature = inspect.signature(self.forward)
            actual_keys = set(key for key in signature.parameters.keys() if key != "self")

            properties = self.parameters["properties"]
            expected_keys = set(properties.keys())

            if actual_keys != expected_keys:
                raise Exception(
                    f"在工具 '{self.name}' 中，'forward' 方法参数是 {actual_keys}，但期望 {expected_keys}。"
                    f"它应该将 'self' 作为第一个参数，然后其下一个参数应该与 tool.parameters['properties'] 的键匹配。"
                )

            json_schema = _convert_type_hints_to_json_schema(self.forward, error_on_missing_type_hints=False)[
                "properties"
            ]  # 此函数不会在缺少文档字符串时引发错误，与 get_json_schema 相反
            for key, value in self.parameters["properties"].items():
                assert key in json_schema, (
                    f"输入 '{key}' 应该存在于函数签名中，只找到 {json_schema.keys()}"
                )
                if "nullable" in value:
                    assert "nullable" in json_schema[key], (
                        f"Nullable argument '{key}' in inputs should have key 'nullable' set to True in function signature."
                    )
                if key in json_schema and "nullable" in json_schema[key]:
                    assert "nullable" in value, (
                        f"Nullable argument '{key}' in function signature should have key 'nullable' set to True in inputs."
                    )

    def forward(self, *args, **kwargs):
        raise NotImplementedError("在您的 `Tool` 子类中编写此方法。")

    def __call__(self, *args, sanitize_inputs_outputs: bool = False, **kwargs):
        if not self.is_initialized:
            self.setup()

        # Handle the arguments might be passed as a single dictionary
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            potential_kwargs = args[0]

            # If the dictionary keys match our input parameters, convert it to kwargs
            if all(key in self.parameters['properties'] for key in potential_kwargs):
                args = ()
                kwargs = potential_kwargs

        outputs = self.forward(*args, **kwargs)

        return outputs

    def setup(self):
        """
        在此处重写此方法，用于任何昂贵且需要在开始使用工具之前执行的操作。例如加载大型模型。
        """
        self.is_initialized = True

    def to_dict(self) -> dict:
        """返回表示工具的字典"""
        class_name = self.__class__.__name__
        if type(self).__name__ == "SimpleTool":
            # 检查导入是否自包含
            source_code = get_source(self.forward).replace("@tool", "")
            forward_node = ast.parse(source_code)
            # 如果工具是使用 '@tool' 装饰器创建的，它只有一个 forward 传递，所以只需获取其代码更简单
            method_checker = MethodChecker(set())
            method_checker.visit(forward_node)

            if len(method_checker.errors) > 0:
                errors = [f"- {error}" for error in method_checker.errors]
                raise (ValueError(f"SimpleTool 验证失败 {self.name}：\n" + "\n".join(errors)))

            forward_source_code = get_source(self.forward)
            tool_code = textwrap.dedent(
                f"""
            from smolagents import Tool
            from typing import Any, Optional

            class {class_name}(Tool):
                name = "{self.name}"
                description = {json.dumps(textwrap.dedent(self.description).strip())}
                inputs = {repr(self.inputs)}
                output_type = "{self.output_type}"
            """
            ).strip()
            import re

            def add_self_argument(source_code: str) -> str:
                """如果不存在，将 'self' 作为第一个参数添加到函数定义中。"""
                pattern = r"def forward\(((?!self)[^)]*)\)"

                def replacement(match):
                    args = match.group(1).strip()
                    if args:  # 如果有其他参数
                        return f"def forward(self, {args})"
                    return "def forward(self)"

                return re.sub(pattern, replacement, source_code)

            forward_source_code = forward_source_code.replace(self.name, "forward")
            forward_source_code = add_self_argument(forward_source_code)
            forward_source_code = forward_source_code.replace("@tool", "").strip()
            tool_code += "\n\n" + textwrap.indent(forward_source_code, "    ")

        else:  # 如果工具不是由 @tool 装饰器创建的，它是通过子类化 Tool 创建的
            if type(self).__name__ in [
                "SpaceToolWrapper",
                "LangChainToolWrapper",
                "GradioToolWrapper",
            ]:
                raise ValueError(
                    "无法保存使用 from_space、from_langchain 或 from_gradio 创建的对象，因为这会产生错误。"
                )

            validate_tool_attributes(self.__class__)

            tool_code = "from typing import Any, Optional\n" + instance_to_source(self, base_cls=Tool)

        requirements = {el for el in get_imports(tool_code) if el not in sys.stdlib_module_names} | {"smolagents"}

        return {"name": self.name, "code": tool_code, "requirements": sorted(requirements)}

    @classmethod
    def from_dict(cls, tool_dict: dict[str, Any], **kwargs) -> "Tool":
        """
        Create tool from a dictionary representation.

        Args:
            tool_dict (`dict[str, Any]`): Dictionary representation of the tool.
            **kwargs: Additional keyword arguments to pass to the tool's constructor.

        Returns:
            `Tool`: Tool object.
        """
        if "code" not in tool_dict:
            raise ValueError("Tool dictionary must contain 'code' key with the tool source code")
        return cls.from_code(tool_dict["code"], **kwargs)

    def save(self, output_dir: str | Path, tool_file_name: str = "tool", make_gradio_app: bool = True):
        """
        Saves the relevant code files for your tool so it can be pushed to the Hub. This will copy the code of your
        tool in `output_dir` as well as autogenerate:

        - a `{tool_file_name}.py` file containing the logic for your tool.
        If you pass `make_gradio_app=True`, this will also write:
        - an `app.py` file providing a UI for your tool when it is exported to a Space with `tool.push_to_hub()`
        - a `requirements.txt` containing the names of the modules used by your tool (as detected when inspecting its
          code)

        Args:
            output_dir (`str` or `Path`): The folder in which you want to save your tool.
            tool_file_name (`str`, *optional*): The file name in which you want to save your tool.
            make_gradio_app (`bool`, *optional*, defaults to True): Whether to also export a `requirements.txt` file and Gradio UI.
        """
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        # Save tool file
        self._write_file(output_path / f"{tool_file_name}.py", self._get_tool_code())
        if make_gradio_app:
            #  Save app file
            self._write_file(output_path / "app.py", self._get_gradio_app_code(tool_module_name=tool_file_name))
            # Save requirements file
            self._write_file(output_path / "requirements.txt", self._get_requirements())

    def _write_file(self, file_path: Path, content: str) -> None:
        """Writes content to a file with UTF-8 encoding."""
        file_path.write_text(content, encoding="utf-8")

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload tool",
        private: bool | None = None,
        token: bool | str | None = None,
        create_pr: bool = False,
    ) -> str:
        """
        将工具上传到 Hub。

        参数:
            repo_id (`str`):
                您想要推送工具到的存储库名称。推送到给定组织时，它应该包含您的组织名称。
            commit_message (`str`, *可选*, 默认为 `"Upload tool"`):
                推送时提交的消息。
            private (`bool`, *可选*):
                是否使存储库私有。如果为 `None`（默认），除非组织的默认值为私有，否则存储库将是公共的。如果存储库已存在，则忽略此值。
            token (`bool` 或 `str`, *可选*):
                用作远程文件的 HTTP bearer 授权的令牌。如果未设置，将使用运行 `huggingface-cli login` 时生成的令牌（存储在 `~/.huggingface` 中）。
            create_pr (`bool`, *可选*, 默认为 `False`):
                是否使用上传的文件创建 PR 或直接提交。
        """
        # 初始化存储库
        repo_id = self._initialize_hub_repo(repo_id, token, private)
        # Prepare files for commit
        additions = self._prepare_hub_files()
        # Create commit
        return create_commit(
            repo_id=repo_id,
            operations=additions,
            commit_message=commit_message,
            token=token,
            create_pr=create_pr,
            repo_type="space",
        )

    @staticmethod
    def _initialize_hub_repo(repo_id: str, token: bool | str | None, private: bool | None) -> str:
        """Initialize repository on Hugging Face Hub."""
        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="space",
            space_sdk="gradio",
        )
        metadata_update(repo_url.repo_id, {"tags": ["smolagents", "tool"]}, repo_type="space", token=token)
        return repo_url.repo_id

    def _prepare_hub_files(self) -> list:
        """准备 Hub 提交的文件。"""
        additions = [
            # 添加工具代码
            CommitOperationAdd(
                path_in_repo="tool.py",
                path_or_fileobj=self._get_tool_code().encode(),
            ),
            # Add Gradio app
            CommitOperationAdd(
                path_in_repo="app.py",
                path_or_fileobj=self._get_gradio_app_code().encode(),
            ),
            # Add requirements
            CommitOperationAdd(
                path_in_repo="requirements.txt",
                path_or_fileobj=self._get_requirements().encode(),
            ),
        ]
        return additions

    def _get_tool_code(self) -> str:
        """获取工具的代码。"""
        return self.to_dict()["code"]

    def _get_gradio_app_code(self, tool_module_name: str = "tool") -> str:
        """Get the Gradio app code."""
        class_name = self.__class__.__name__
        return textwrap.dedent(
            f"""\
            from smolagents import launch_gradio_demo
            from {tool_module_name} import {class_name}

            tool = {class_name}()
            launch_gradio_demo(tool)
            """
        )

    def _get_requirements(self) -> str:
        """获取依赖项。"""
        return "\n".join(self.to_dict()["requirements"])

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        token: str | None = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        加载在 Hub 上定义的工具。

        <Tip warning={true}>

        从 Hub 加载工具意味着您将下载工具并在本地执行它。
        在运行时加载之前，请始终检查您正在下载的工具，就像使用 pip/npm/apt 安装包时一样。

        </Tip>

        参数:
            repo_id (`str`):
                在 Hub 上定义工具的 Space 存储库名称。
            token (`str`, *可选*):
                用于在 hf.co 上识别您的令牌。如果未设置，将使用运行 `huggingface-cli login` 时生成的令牌（存储在 `~/.huggingface` 中）。
            trust_remote_code(`str`, *可选*, 默认为 False):
                此标志表示您了解运行远程代码的风险并且您信任此工具。
                如果不将其设置为 True，从 Hub 加载工具将失败。
            kwargs (附加关键字参数, *可选*):
                将分为两部分的附加关键字参数：与 Hub 相关的所有参数（如 `cache_dir`, `revision`, `subfolder`）将在下载工具文件时使用，其他参数将传递给其 init。
        """
        if not trust_remote_code:
            raise ValueError(
                "从 Hub 加载工具需要确认您信任其代码：为此，请传递 `trust_remote_code=True`。"
            )

        # 获取工具的 tool.py 文件。
        tool_file = hf_hub_download(
            repo_id,
            "tool.py",
            token=token,
            repo_type="space",
            cache_dir=kwargs.get("cache_dir"),
            force_download=kwargs.get("force_download"),
            proxies=kwargs.get("proxies"),
            revision=kwargs.get("revision"),
            subfolder=kwargs.get("subfolder"),
            local_files_only=kwargs.get("local_files_only"),
        )

        tool_code = Path(tool_file).read_text()
        return Tool.from_code(tool_code, **kwargs)

    @classmethod
    def from_code(cls, tool_code: str, **kwargs):
        module = types.ModuleType("dynamic_tool")

        exec(tool_code, module.__dict__)

        # Find the Tool subclass
        tool_class = next(
            (
                obj
                for _, obj in inspect.getmembers(module, inspect.isclass)
                if issubclass(obj, Tool) and obj is not Tool
            ),
            None,
        )

        if tool_class is None:
            raise ValueError("在代码中未找到 Tool 子类。")

        # 处理向后兼容性：如果代码使用 'inputs' 属性而不是 'parameters'
        # 检查 inputs 是否在类字典中定义（不是通过属性）
        class_dict = tool_class.__dict__
        if 'inputs' in class_dict and 'parameters' not in class_dict:
            # Old code format with inputs, need to convert to parameters
            inputs_value = class_dict['inputs']
            if isinstance(inputs_value, str):
                inputs_value = ast.literal_eval(inputs_value)
            tool_class.parameters = {
                "type": "object",
                "properties": inputs_value,
            }

        return tool_class(**kwargs)

    @staticmethod
    def from_space(
        space_id: str,
        name: str,
        description: str,
        api_name: str | None = None,
        token: str | None = None,
    ):
        """
        根据 Hub 上的 Space ID 从 Space 创建 [`Tool`]。

        参数:
            space_id (`str`):
                Hub 上 Space 的 ID。
            name (`str`):
                工具的名称。
            description (`str`):
                工具的描述。
            api_name (`str`, *可选*):
                要使用的特定 api_name，如果 space 有多个标签页。如果未指定，将默认为第一个可用的 api。
            token (`str`, *可选*):
                添加您的令牌以访问私有空间或增加 GPU 配额。
        返回:
            [`Tool`]:
                Space，作为工具。

        示例:
        ```py
        >>> web_searcher = Tool.from_space(
        ...     space_id="example/research-tool",
        ...     name="web-searcher",
        ...     description="搜索网络信息"
        ... )
        >>> results = web_searcher("搜索 Python 教程")
        ```
        ```py
        >>> face_swapper = Tool.from_space(
        ...     "tuan2308/face-swap",
        ...     "face_swapper",
        ...     "将第一张图像中显示的脸放在第二张图像上的工具。您可以给它图像路径。",
        ... )
        >>> image = face_swapper('./aymeric.jpeg', './ruth.jpg')
        ```
        """
        from gradio_client import Client, handle_file

        class SpaceToolWrapper(Tool):
            skip_forward_signature_validation = True

            def __init__(
                self,
                space_id: str,
                name: str,
                description: str,
                api_name: str | None = None,
                token: str | None = None,
            ):
                self.name = name
                self.description = description
                self.client = Client(space_id, hf_token=token)
                space_description = self.client.view_api(return_format="dict", print_info=False)["named_endpoints"]

                # If api_name is not defined, take the first of the available APIs for this space
                if api_name is None:
                    api_name = list(space_description.keys())[0]
                    logger.warning(
                        f"Since `api_name` was not defined, it was automatically set to the first available API: `{api_name}`."
                    )
                self.api_name = api_name

                try:
                    space_description_api = space_description[api_name]
                except KeyError:
                    raise KeyError(f"在可用的 api 名称中找不到指定的 {api_name=}。")

                properties = {}
                for parameter in space_description_api["parameters"]:
                    if not parameter["parameter_has_default"]:
                        parameter_type = parameter["type"]["type"]
                        if parameter_type == "object":
                            parameter_type = "any"
                        properties[parameter["parameter_name"]] = {
                            "type": parameter_type,
                            "description": parameter["python_type"]["description"],
                        }
                self.parameters = {
                    "type": "object",
                    "properties": properties,
                }
                output_component = space_description_api["returns"][0]["component"]
                if output_component == "Image":
                    self.output_type = "image"
                elif output_component == "Audio":
                    self.output_type = "audio"
                else:
                    self.output_type = "any"
                self.is_initialized = True

            def sanitize_argument_for_prediction(self, arg):
                from gradio_client.utils import is_http_url_like
                from PIL.Image import Image

                if isinstance(arg, Image):
                    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    arg.save(temp_file.name)
                    arg = temp_file.name
                if (
                    (isinstance(arg, str) and os.path.isfile(arg))
                    or (isinstance(arg, Path) and arg.exists() and arg.is_file())
                    or is_http_url_like(arg)
                ):
                    arg = handle_file(arg)
                return arg

            def forward(self, *args, **kwargs):
                # 预处理 args 和 kwargs：
                args = list(args)
                for i, arg in enumerate(args):
                    args[i] = self.sanitize_argument_for_prediction(arg)
                for arg_name, arg in kwargs.items():
                    kwargs[arg_name] = self.sanitize_argument_for_prediction(arg)

                output = self.client.predict(*args, api_name=self.api_name, **kwargs)
                if isinstance(output, tuple) or isinstance(output, list):
                    return output[
                        0
                    ]  # 有时 space 也会返回生成种子，在这种情况下结果在索引 0 处
                return output

        return SpaceToolWrapper(
            space_id=space_id,
            name=name,
            description=description,
            api_name=api_name,
            token=token,
        )

    @staticmethod
    def from_gradio(gradio_tool):
        """
        从 gradio 工具创建 [`Tool`]。
        """
        import inspect

        class GradioToolWrapper(Tool):
            def __init__(self, _gradio_tool):
                self.name = _gradio_tool.name
                self.description = _gradio_tool.description
                self.output_type = "string"
                self._gradio_tool = _gradio_tool
                func_args = list(inspect.signature(_gradio_tool.run).parameters.items())
                properties = {
                    key: {"type": CONVERSION_DICT[value.annotation], "description": ""} for key, value in func_args
                }
                self.parameters = {
                    "type": "object",
                    "properties": properties,
                }
                self.forward = self._gradio_tool.run

        return GradioToolWrapper(gradio_tool)

    @staticmethod
    def from_langchain(langchain_tool):
        """
        从 langchain 工具创建 [`Tool`]。
        """

        class LangChainToolWrapper(Tool):
            skip_forward_signature_validation = True

            def __init__(self, _langchain_tool):
                self.name = _langchain_tool.name.lower()
                self.description = _langchain_tool.description
                properties = _langchain_tool.args.copy()
                for input_content in properties.values():
                    if "title" in input_content:
                        input_content.pop("title")
                    input_content["description"] = ""
                self.parameters = {
                    "type": "object",
                    "properties": properties,
                }
                self.output_type = "string"
                self.langchain_tool = _langchain_tool
                self.is_initialized = True

            def forward(self, *args, **kwargs):
                tool_input = kwargs.copy()
                properties = self.parameters.get("properties", {})
                for index, argument in enumerate(args):
                    if index < len(properties):
                        input_key = next(iter(properties))
                        tool_input[input_key] = argument
                return self.langchain_tool.run(tool_input)

        return LangChainToolWrapper(langchain_tool)
    
    
class AsyncTool(Tool):
    async def forward(self, *args, **kwargs):
        raise NotImplementedError("在您的 `AsyncTool` 子类中编写此方法。")

    async def __call__(self, *args, sanitize_inputs_outputs: bool = False, **kwargs):
        if not self.is_initialized:
            self.setup()

        # Handle the arguments might be passed as a single dictionary
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            potential_kwargs = args[0]

            # If the dictionary keys match our input parameters, convert it to kwargs
            if all(key in self.parameters["properties"] for key in potential_kwargs):
                args = ()
                kwargs = potential_kwargs

        outputs = await self.forward(*args, **kwargs)

        return outputs


def launch_gradio_demo(tool: Tool):
    """
    为工具启动 gradio 演示。相应的工具类需要正确实现类属性 `inputs` 和 `output_type`。

    参数:
        tool (`Tool`): 要为其启动演示的工具。
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("应该安装 Gradio 以启动 gradio 演示。")

    TYPE_TO_COMPONENT_CLASS_MAPPING = {
        "boolean": gr.Checkbox,
        "image": gr.Image,
        "audio": gr.Audio,
        "string": gr.Textbox,
        "integer": gr.Textbox,
        "number": gr.Textbox,
    }

    def tool_forward(*args, **kwargs):
        return tool(*args, sanitize_inputs_outputs=True, **kwargs)

    tool_forward.__signature__ = inspect.signature(tool.forward)

    gradio_inputs = []
    for input_name, input_details in tool.parameters.get("properties", {}).items():
        input_gradio_component_class = TYPE_TO_COMPONENT_CLASS_MAPPING[input_details["type"]]
        new_component = input_gradio_component_class(label=input_name)
        gradio_inputs.append(new_component)

    output_gradio_component_class = TYPE_TO_COMPONENT_CLASS_MAPPING[tool.output_type]
    gradio_output = output_gradio_component_class(label="Output")

    gr.Interface(
        fn=tool_forward,
        inputs=gradio_inputs,
        outputs=gradio_output,
        title=tool.name,
        description=tool.description,
        api_name=tool.name,
    ).launch()


def load_tool(
    repo_id,
    model_repo_id: Optional[str] = None,
    token: Optional[str] = None,
    trust_remote_code: bool = False,
    **kwargs,
):
    """
    从 Hub 快速加载工具的主函数。

    <Tip warning={true}>

    加载工具意味着您将下载工具并在本地执行它。
    在运行时加载之前，请始终检查您正在下载的工具，就像使用 pip/npm/apt 安装包时一样。

    </Tip>

    参数:
        repo_id (`str`):
            Hub 上工具的 Space 存储库 ID。
        model_repo_id (`str`, *可选*):
            使用此参数为所选工具使用与默认模型不同的模型。
        token (`str`, *可选*):
            用于在 hf.co 上识别您的令牌。如果未设置，将使用运行 `huggingface-cli login` 时生成的令牌（存储在 `~/.huggingface` 中）。
        trust_remote_code (`bool`, *可选*, 默认为 False):
            需要接受此参数才能从 Hub 加载工具。
        kwargs (附加关键字参数, *可选*):
            将分为两部分的附加关键字参数：与 Hub 相关的所有参数（如 `cache_dir`, `revision`, `subfolder`）将在下载工具文件时使用，其他参数将传递给其 init。
    """
    return Tool.from_hub(
        repo_id,
        model_repo_id=model_repo_id,
        token=token,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )


def add_description(description):
    """
    向函数添加描述的装饰器。
    """

    def inner(func):
        func.description = description
        func.name = func.__name__
        return func

    return inner


class ToolCollection:
    """
    工具集合允许在智能体的工具箱中加载工具集合。

    集合可以从 Hub 中的集合或 MCP 服务器加载，请参阅：
    - [`ToolCollection.from_hub`]
    - [`ToolCollection.from_mcp`]

    有关示例和用法，请参阅：[`ToolCollection.from_hub`] 和 [`ToolCollection.from_mcp`]
    """

    def __init__(self, tools: List[Tool]):
        self.tools = tools

    @classmethod
    def from_hub(
        cls,
        collection_slug: str,
        token: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> "ToolCollection":
        """从 Hub 加载工具集合。

        它将集合中所有 Spaces 的工具集合添加到智能体的工具箱中

        > [!NOTE]
        > 只会获取 Spaces，因此如果您希望此集合展示它们，可以随意将模型和数据集添加到您的集合中。

        参数:
            collection_slug (str): 引用集合的集合 slug。
            token (str, *可选*): 如果集合是私有的，则为身份验证令牌。
            trust_remote_code (bool, *可选*, 默认为 False): 是否信任远程代码。

        返回:
            ToolCollection: 加载了工具的工具集合实例。

        示例:
        ```py
        >>> from smolagents import ToolCollection

        >>> image_tool_collection = ToolCollection.from_hub("huggingface-tools/diffusion-tools-6630bb19a942c2306a2cdb6f")
        >>> # agent = SomeAgent(tools=[*image_tool_collection.tools], add_base_tools=True)

        >>> await agent.run("请为我画一幅河流和湖泊的图片。")
        ```
        """
        _collection = get_collection(collection_slug, token=token)
        _hub_repo_ids = {item.item_id for item in _collection.items if item.item_type == "space"}

        tools = {Tool.from_hub(repo_id, token, trust_remote_code) for repo_id in _hub_repo_ids}

        return cls(tools)

    @classmethod
    @contextmanager
    def from_mcp(
        cls, server_parameters: Union["mcp.StdioServerParameters", dict], trust_remote_code: bool = False
    ) -> "ToolCollection":
        """自动从 MCP 服务器加载工具集合。

        此方法支持 SSE 和 Stdio MCP 服务器。查看 `server_parameters`
        参数以获取有关如何连接到 SSE 或 Stdio MCP 服务器的更多详细信息。

        注意：将生成一个单独的线程来运行处理 MCP 服务器的 asyncio 事件循环。

        参数:
            server_parameters (`mcp.StdioServerParameters` 或 `dict`):
                用于连接到 MCP 服务器的服务器参数。如果提供字典，
                则假定它是 `mcp.client.sse.sse_client` 的参数。
            trust_remote_code (`bool`, *可选*, 默认为 `False`):
                是否信任在 MCP 服务器上定义的工具的代码执行。
                只有在您信任 MCP 服务器并了解在本地机器上运行远程代码的风险时，才应将此选项设置为 `True`。
                如果设置为 `False`，从 MCP 加载工具将失败。


        返回:
            ToolCollection: 工具集合实例。

        使用 Stdio MCP 服务器的示例:
        ```py
        >>> from smolagents import ToolCollection
        >>> from mcp import StdioServerParameters

        >>> server_parameters = StdioServerParameters(
        >>>     command="uv",
        >>>     args=["--quiet", "pubmedmcp@0.1.3"],
        >>>     env={"UV_PYTHON": "3.12", **os.environ},
        >>> )

        >>> with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
        >>>     # agent = SomeAgent(tools=[*tool_collection.tools], add_base_tools=True)
        >>>     # await agent.run("请找到治疗宿醉的方法。")
        ```

        使用 SSE MCP 服务器的示例:
        ```py
        >>> with ToolCollection.from_mcp({"url": "http://127.0.0.1:8000/sse"}, trust_remote_code=True) as tool_collection:
        >>>     # agent = SomeAgent(tools=[*tool_collection.tools], add_base_tools=True)
        >>>     # await agent.run("请找到治疗宿醉的方法。")
        ```
        """
        if not trust_remote_code:
            raise ValueError(
                "从 MCP 加载工具需要您确认信任 MCP 服务器，"
                "因为它将在您的本地机器上执行代码：传递 `trust_remote_code=True`。"
            )
        try:
            from mcpadapt.core import MCPAdapt
            from mcpadapt.smolagents_adapter import SmolAgentsAdapter
        except ImportError:
            raise ImportError(
                """请安装 'mcp' 额外依赖以使用 ToolCollection.from_mcp：`pip install "smolagents[mcp]"`。"""
            )

        with MCPAdapt(server_parameters, SmolAgentsAdapter()) as tools:
            yield cls(tools)


def tool(tool_function: Callable) -> Tool:
    """
    将函数转换为动态创建的 Tool 子类的实例。

    参数:
        tool_function (`Callable`): 要转换为 Tool 子类的函数。
            应该为每个输入提供类型提示，并为输出提供类型提示。
            还应该有一个包含函数描述的文档字符串
            和一个 'Args:' 部分，其中描述每个参数。
    """
    tool_json_schema = get_json_schema(tool_function)["function"]
    if "return" not in tool_json_schema:
        raise TypeHintParsingException("Tool return type not found: make sure your function has a return type hint!")

    class SimpleTool(Tool):
        def __init__(self):
            self.is_initialized = True

    # Set the class attributes
    SimpleTool.name = tool_json_schema["name"]
    SimpleTool.description = tool_json_schema["description"]
    SimpleTool.parameters = tool_json_schema["parameters"]
    SimpleTool.output_type = tool_json_schema["return"]["type"]

    @wraps(tool_function)
    def wrapped_function(*args, **kwargs):
        return tool_function(*args, **kwargs)

    # Bind the copied function to the forward method
    SimpleTool.forward = staticmethod(wrapped_function)

    # Get the signature parameters of the tool function
    sig = inspect.signature(tool_function)
    # - Add "self" as first parameter to tool_function signature
    new_sig = sig.replace(
        parameters=[inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)] + list(sig.parameters.values())
    )
    # - Set the signature of the forward method
    SimpleTool.forward.__signature__ = new_sig

    # Create and attach the source code of the dynamically created tool class and forward method
    # - Get the source code of tool_function
    tool_source = inspect.getsource(tool_function)
    # - Remove the tool decorator and function definition line
    tool_source_body = "\n".join(tool_source.split("\n")[2:])
    # - Dedent
    tool_source_body = textwrap.dedent(tool_source_body)
    # - Create the forward method source, including def line and indentation
    forward_method_source = f"def forward{str(new_sig)}:\n{textwrap.indent(tool_source_body, '    ')}"
    # - Create the class source
    class_source = (
            textwrap.dedent(f'''
            class SimpleTool(Tool):
                name: str = "{tool_json_schema["name"]}"
                description: str = {json.dumps(textwrap.dedent(tool_json_schema["description"]).strip())}
                parameters: dict[str, dict[str, str]] = {json.dumps(tool_json_schema["parameters"])}
                output_type: str = "{tool_json_schema["return"]["type"]}"

            def __init__(self):
                self.is_initialized = True

        ''')
        + textwrap.indent(forward_method_source, "    ")  # indent for class method
    )
    # - Store the source code on both class and method for inspection
    SimpleTool.__source__ = class_source
    SimpleTool.forward.__source__ = forward_method_source

    simple_tool = SimpleTool()
    return simple_tool


class PipelineTool(Tool):
    """
    针对 Transformer 模型定制的 [`Tool`]。除了基类 [`Tool`] 的类属性外，您还需要指定：

    - **model_class** (`type`) -- 用于在此工具中加载模型的类。
    - **default_checkpoint** (`str`) -- 当用户未指定时应该使用的默认检查点。
    - **pre_processor_class** (`type`, *可选*, 默认为 [`transformers.AutoProcessor`]) -- 用于加载预处理器的类。
    - **post_processor_class** (`type`, *可选*, 默认为 [`transformers.AutoProcessor`]) -- 用于加载后处理器的类（当与预处理器不同时）。

    参数:
        model (`str` 或 [`transformers.PreTrainedModel`], *可选*):
            用于模型的检查点名称，或实例化的模型。如果未设置，将默认为类属性 `default_checkpoint` 的值。
        pre_processor (`str` 或 `Any`, *可选*):
            用于预处理器的检查点名称，或实例化的预处理器（可以是 tokenizer、图像处理器、特征提取器或处理器）。如果未设置，将默认为 `model` 的值。
        post_processor (`str` 或 `Any`, *可选*):
            用于后处理器的检查点名称，或实例化的预处理器（可以是 tokenizer、图像处理器、特征提取器或处理器）。如果未设置，将默认为 `pre_processor`。
        device (`int`, `str` 或 `torch.device`, *可选*):
            执行模型的设备。将默认为任何可用的加速器（GPU、MPS 等），否则为 CPU。
        device_map (`str` 或 `dict`, *可选*):
            如果传递，将用于实例化模型。
        model_kwargs (`dict`, *可选*):
            发送到模型实例化的任何关键字参数。
        token (`str`, *可选*):
            用作远程文件的 HTTP bearer 授权的令牌。如果未设置，将使用运行 `huggingface-cli login` 时生成的令牌（存储在 `~/.huggingface` 中）。
        hub_kwargs (附加关键字参数, *可选*):
            发送到将从 Hub 加载数据的方法的任何附加关键字参数。
    """

    pre_processor_class = None
    model_class = None
    post_processor_class = None
    default_checkpoint = None
    description = "This is a pipeline tool"
    name = "pipeline"
    inputs = {"prompt": str}
    output_type = str
    skip_forward_signature_validation = True

    def __init__(
        self,
        model=None,
        pre_processor=None,
        post_processor=None,
        device=None,
        device_map=None,
        model_kwargs=None,
        token=None,
        **hub_kwargs,
    ):
        if not _is_package_available("accelerate") or not _is_package_available("torch"):
            raise ModuleNotFoundError(
                "请安装 'transformers' 额外依赖以使用 PipelineTool：`pip install 'smolagents[transformers]'`"
            )

        if model is None:
            if self.default_checkpoint is None:
                raise ValueError("This tool does not implement a default checkpoint, you need to pass one.")
            model = self.default_checkpoint
        if pre_processor is None:
            pre_processor = model

        self.model = model
        self.pre_processor = pre_processor
        self.post_processor = post_processor
        self.device = device
        self.device_map = device_map
        self.model_kwargs = {} if model_kwargs is None else model_kwargs
        if device_map is not None:
            self.model_kwargs["device_map"] = device_map
        self.hub_kwargs = hub_kwargs
        self.hub_kwargs["token"] = token

        super().__init__()

    def setup(self):
        """
        如有必要，实例化 `pre_processor`、`model` 和 `post_processor`。
        """
        if isinstance(self.pre_processor, str):
            if self.pre_processor_class is None:
                from transformers import AutoProcessor

                self.pre_processor_class = AutoProcessor
            self.pre_processor = self.pre_processor_class.from_pretrained(self.pre_processor, **self.hub_kwargs)

        if isinstance(self.model, str):
            self.model = self.model_class.from_pretrained(self.model, **self.model_kwargs, **self.hub_kwargs)

        if self.post_processor is None:
            self.post_processor = self.pre_processor
        elif isinstance(self.post_processor, str):
            if self.post_processor_class is None:
                from transformers import AutoProcessor

                self.post_processor_class = AutoProcessor
            self.post_processor = self.post_processor_class.from_pretrained(self.post_processor, **self.hub_kwargs)

        if self.device is None:
            if self.device_map is not None:
                self.device = list(self.model.hf_device_map.values())[0]
            else:
                from accelerate import PartialState

                self.device = PartialState().default_device

        if self.device_map is None:
            self.model.to(self.device)

        super().setup()

    def encode(self, raw_inputs):
        """
        Uses the `pre_processor` to prepare the inputs for the `model`.
        """
        return self.pre_processor(raw_inputs)

    def forward(self, inputs):
        """
        Sends the inputs through the `model`.
        """
        import torch

        with torch.no_grad():
            return self.model(**inputs)

    def decode(self, outputs):
        """
        Uses the `post_processor` to decode the model output.
        """
        return self.post_processor(outputs)

    def __call__(self, *args, sanitize_inputs_outputs: bool = False, **kwargs):
        import torch
        from accelerate.utils import send_to_device

        if not self.is_initialized:
            self.setup()

        if sanitize_inputs_outputs:
            args, kwargs = handle_agent_input_types(*args, **kwargs)
        encoded_inputs = self.encode(*args, **kwargs)

        tensor_inputs = {k: v for k, v in encoded_inputs.items() if isinstance(v, torch.Tensor)}
        non_tensor_inputs = {k: v for k, v in encoded_inputs.items() if not isinstance(v, torch.Tensor)}

        encoded_inputs = send_to_device(tensor_inputs, self.device)
        outputs = self.forward({**encoded_inputs, **non_tensor_inputs})
        outputs = send_to_device(outputs, "cpu")
        decoded_outputs = self.decode(outputs)
        if sanitize_inputs_outputs:
            decoded_outputs = handle_agent_output_types(decoded_outputs, self.output_type)
        return decoded_outputs


def get_tools_definition_code(tools: Dict[str, Tool]) -> str:
    tool_codes = []
    for tool in tools.values():
        validate_tool_attributes(tool.__class__, check_imports=False)
        tool_code = instance_to_source(tool, base_cls=Tool)
        tool_code = tool_code.replace("from smolagents.tools import Tool", "")
        tool_code += f"\n\n{tool.name} = {tool.__class__.__name__}()\n"
        tool_codes.append(tool_code)

    tool_definition_code = "\n".join([f"import {module}" for module in BASE_BUILTIN_MODULES])
    tool_definition_code += textwrap.dedent(
        """
    from typing import Any

    class Tool:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, *args, **kwargs):
            pass # to be implemented in child class
    """
    )
    tool_definition_code += "\n\n".join(tool_codes)
    return tool_definition_code


def make_tool_instance(agent):
    agnet_name = agent.name
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "any",
                "description": "要由团队成员执行的任务。",
            },
        },
        "required": ["task"],
    }
    output_type = "any"
    async def forward(self, task: Any) -> ToolResult:
        result = await agent.run(task)
        return ToolResult(output=result, error=None)

    tool_cls = type(
        f"{agnet_name}",
        (AsyncTool,),
        {
            "name": agnet_name,
            "description": agent.description,
            "parameters": parameters,
            "output_type": output_type,
            "forward": forward,
        }
    )

    tool_instance = tool_cls()

    return tool_instance

__all__ = [
    "AUTHORIZED_TYPES",
    "Tool",
    "AsyncTool",
    "tool",
    "load_tool",
    "launch_gradio_demo",
    "ToolCollection",
    "make_tool_instance"
]
