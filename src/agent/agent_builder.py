from typing import Dict, Any, Optional

from src.registry import AGENT, TOOL, Registry
from src.models import model_manager
from src.logger import logger
from src.config import Config

async def build_agent(
    config: Config,
    agent_config: Dict[str, Any],
    default_tools: Optional[Registry] = None,
    default_mcp_tools: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    基于提供的配置构建智能体。

    参数:
        config (dict): 包含工具和模型设置的配置字典。
        agent_config (dict): 包含智能体设置的配置字典。

    返回:
        智能体实例。
    """
    tools = []
    mcp_tools = []
    # 构建工具
    if default_tools is None:
                logger.info("| 未提供默认工具。跳过工具初始化。")
    else:
        used_tools = agent_config.get("tools", [])
        for tool_name in used_tools:
            if tool_name not in default_tools:
                logger.warning(f"工具 '{tool_name}' 未注册。跳过。")
                continue  # 跳过未注册的工具
            config_name = f"{tool_name}_config"  # 例如："python_interpreter_tool" -> "python_interpreter_tool_config"
            if config_name in config:
                # 如果工具有特定配置，使用它
                tool_config = config[config_name]
            else:
                # 否则，使用默认工具实例
                tool_config = dict(type=tool_name)
            tool = TOOL.build(tool_config)
            tools.append(tool)
        logger.info(f"| 工具已初始化: {', '.join([tool.name for tool in tools])}")

    # 构建 MCP 工具
    if default_mcp_tools is None:
        logger.info("| 未提供 MCP 工具。跳过 MCP 工具初始化。")
    else:
        used_mcp_tools = agent_config.get("mcp_tools", [])
        for tools_name in used_mcp_tools:
            if tools_name not in default_mcp_tools:
                logger.warning(f"MCP 工具 '{tools_name}' 不可用。跳过。")
            else:
                mcp_tool = default_mcp_tools[tools_name]
                mcp_tools.append(mcp_tool)
        logger.info(f"| MCP 工具已初始化: {', '.join([tool.name for tool in mcp_tools])}")

    # 加载模型
    model_id = agent_config.get("model_id")
    if not model_id:
        raise ValueError("agent_config 必须包含 'model_id'")
    if model_id not in model_manager.registered_models:
        available_models = ', '.join(model_manager.registered_models.keys())
        raise ValueError(
            f"模型 '{model_id}' 未注册。"
            f"可用模型: {available_models}"
        )
    model = model_manager.registered_models[model_id]

    # 构建智能体
    combined_tools = tools + mcp_tools
    # agent_config 是来自 YAML 配置的字典
    agent_build_config = dict(
        type=agent_config.get('type'),
        config=agent_config,
        model=model,
        tools=combined_tools,
        max_steps=agent_config.get('max_steps'),
        name=agent_config.get('name'),
        description=agent_config.get('description')
    )
    agent = AGENT.build(agent_build_config)

    return agent