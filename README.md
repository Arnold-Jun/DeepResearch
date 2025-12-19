# DeepResearch

<div align="center">

**一个强大的多智能体协作研究系统**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

</div>

## 📖 简介

DeepResearch 是一个基于多智能体协作的深度研究系统，能够自动分解复杂任务、并行执行子任务，并整合结果生成全面的研究报告。系统采用分层编排架构，支持多种专业化的智能体类型，适用于网络搜索、数据分析、网页交互等多种研究场景。

## ✨ 核心特性

- 🤖 **多智能体协作**：支持三种专业化的智能体类型，各司其职
- 🔄 **智能任务分解**：自动将复杂任务分解为可执行的子任务
- ⚡ **并行执行**：支持多个子任务并行执行，提高效率
- 🧠 **多模型支持**：支持配置不同的 LLM 模型用于不同角色
- 🛠️ **丰富的工具生态**：内置网络搜索、浏览器自动化、Python 执行器等工具
- 📊 **失败重试机制**：智能的失败检测和重试策略
- 📝 **自动总结**：自动整合所有子任务结果，生成全面报告
- ⚙️ **灵活配置**：基于 YAML 的配置系统，支持继承和覆盖

## 🏗️ 系统架构

### 整体架构

```
用户任务
  ↓
TopLevelOrchestrator (顶层编排器)
  ├── Planning (规划模块) - 维护父任务列表
  ├── Scheduler (调度模块) - 生成子任务并选择 Agent
  ├── RoundRunner (执行模块) - 并行执行子任务
  ├── Aggregator (聚合模块) - 汇总执行结果
  └── Summarizer (总结模块) - 生成最终答案
  ↓
多种 Agent 类型 (并行执行)
  ├── DeepResearcherAgent - 深度网络搜索
  ├── DeepAnalyzerAgent - 系统性分析
  └── BrowserUseAgent - 浏览器自动化
  ↓
工具系统
  ├── WebSearcherTool - 网络搜索
  ├── PythonInterpreterTool - Python 执行
  ├── AutoBrowserUseTool - 浏览器控制
  └── MCP Tools - 外部工具集成
```

### 三种智能体类型

#### 1. DeepResearcherAgent (深度研究智能体)
- **用途**：深度网络搜索和学术研究
- **工作流**：优化查询 → 搜索网络 → 提取洞察 → 生成后续问题 → 总结
- **适用场景**：需要多轮深度搜索、学术论文查找、广泛信息收集

#### 2. DeepAnalyzerAgent (深度分析智能体)
- **用途**：系统性分析、逐步推理和结构化分析
- **工作流**：分析 → 并行分析（多模型） → 总结
- **适用场景**：数据分析、逻辑推理、结构化分析、逐步解决问题
- **特色**：支持多模型并行分析，最后汇总结果

#### 3. BrowserUseAgent (浏览器使用智能体)
- **用途**：浏览器自动化和网页交互
- **工作流**：调用浏览器工具 → 提取结果
- **适用场景**：网页搜索、浏览特定网站、提取网页内容、与网页元素交互

## 🚀 快速开始

### 环境要求

- Python 3.8+
- pip 或 conda

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/yourusername/DeepResearch.git
cd DeepResearch
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境变量**

在项目根目录创建 `.env` 文件并配置必要的 API 密钥：

```env
# LLM API 配置（根据使用的模型服务配置）
# 注意：根据你实际使用的模型服务，配置相应的 API 密钥
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# 搜索引擎配置（可选，根据配置的搜索引擎选择）
FIRECRAWL_API_KEY=your_firecrawl_api_key
SERPER_API_KEY=your_serper_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_cse_id
BING_API_KEY=your_bing_api_key
```

> **注意**：`.env` 文件已包含在 `.gitignore` 中，不会被提交到仓库。

4. **运行示例**

```bash
# 使用命令行参数
python main.py --config configs/config_main.yaml --task "研究人工智能的最新发展"

# 或交互式运行
python main.py
# 然后输入任务描述
```

## 📝 使用指南

### 基本用法

```bash
python main.py --config configs/config_main.yaml --task "你的研究任务"
```

### 命令行参数

- `--config`: 配置文件路径（默认: `configs/config_main.yaml`）
- `--task`: 要执行的任务描述
- `--max-steps`: 最大执行步数（覆盖配置文件中的设置）

### 配置文件说明

系统使用 YAML 格式的配置文件，支持配置继承。

#### 主配置文件 (`configs/config_main.yaml`)

```yaml
# 继承基础配置
_base_: ./base.yaml

# 通用配置
tag: main
workdir: workdir
log_path: log.txt
save_path: dra.jsonl

# 编排器配置
orchestrator_config:
  max_rounds: 12                    # 最大轮次数
  deadline_seconds: 600             # 全局超时时间（秒）
  max_parallelism: 5                 # 最大并行任务数
  max_failures_per_parent: 6         # 每个父任务的最大失败次数
  subtask_timeout_seconds: 240      # 子任务超时时间（秒）
  subtask_output_max_chars: 1800    # 子任务输出最大字符数
  subtask_failure_threshold: 0.5    # 子任务失败率阈值
  planning_model_id: qwen3-8b       # Planner 使用的模型
  scheduler_model_id: qwen3-8b      # Scheduler 使用的模型
  summarizer_model_id: qwen3-8b     # Summarizer 使用的模型

# 智能体配置
deep_researcher_agent_config:
  type: deep_researcher_agent
  model_id: qwen3-8b
  max_steps: 3
  tools:
    - python_interpreter_tool

deep_analyzer_agent_config:
  type: deep_analyzer_agent
  model_id: qwen3-8b
  max_steps: 3
  analyzer_model_ids:
    - qwen3-8b
  tools:
    - python_interpreter_tool

browser_use_agent_config:
  type: browser_use_agent
  model_id: qwen3-8b
  max_steps: 5
  tools:
    - auto_browser_use_tool
    - python_interpreter_tool
```

#### 基础配置文件 (`configs/base.yaml`)

```yaml
# 工具配置
web_searcher_tool_config:
  type: web_searcher_tool
  engine: Firecrawl  # 选项: "Firecrawl", "Google", "Bing", "DuckDuckGo", "Baidu"
  num_results: 5
  fetch_content: true

auto_browser_use_tool_config:
  type: auto_browser_use_tool
  model_id: qwen3-8b
```

### 工作流程示例

1. **用户输入任务**：例如 "研究量子计算的最新进展"

2. **规划阶段**：Planner 将任务分解为多个父任务
   - 父任务1：搜索量子计算的基础理论
   - 父任务2：查找最新的量子计算研究论文
   - 父任务3：分析量子计算的应用场景

3. **调度阶段**：Scheduler 为每个父任务生成子任务并选择 Agent
   - 父任务1 → 3个子任务 → 使用 `deep_researcher_agent`
   - 父任务2 → 2个子任务 → 使用 `browser_use_agent`
   - 父任务3 → 2个子任务 → 使用 `deep_analyzer_agent`

4. **执行阶段**：RoundRunner 并行执行所有子任务

5. **聚合阶段**：Aggregator 汇总每轮执行结果

6. **总结阶段**：Summarizer 整合所有结果，生成最终报告

## 📁 项目结构

```
DeepResearch/
├── main.py                 # 主入口文件
├── requirements.txt        # 依赖列表
├── README.md              # 项目说明文档
├── configs/               # 配置文件目录
│   ├── base.yaml          # 基础配置
│   └── config_main.yaml   # 主配置
└── src/                   # 源代码目录
    ├── agent/             # 智能体模块
    │   ├── agent_builder.py
    │   ├── deep_researcher_agent/  # 深度研究智能体
    │   ├── deep_analyzer_agent/    # 深度分析智能体
    │   ├── browser_use_agent/      # 浏览器使用智能体
    │   └── common/                 # 通用组件
    ├── orchestrator/      # 编排器模块
    │   ├── orchestrator.py        # 顶层编排器
    │   ├── planner.py              # 规划模块
    │   ├── scheduler.py            # 调度模块
    │   ├── runner.py               # 执行模块
    │   ├── aggregator.py           # 聚合模块
    │   ├── summarizer.py           # 总结模块
    │   └── state.py                # 状态定义
    ├── tools/             # 工具模块
    │   ├── research/      # 研究工具
    │   ├── analysis/      # 分析工具
    │   ├── browser/       # 浏览器工具
    │   └── python_interpreter.py
    ├── models/            # 模型管理
    ├── config/            # 配置管理
    ├── logger/            # 日志系统
    ├── mcp/               # MCP 工具支持
    └── registry.py        # 注册表系统
```

## 🔧 高级配置

### 模型配置

系统支持配置不同的 LLM 模型用于不同角色。在配置文件中可以指定：

- `planning_model_id`: Planner 使用的模型
- `scheduler_model_id`: Scheduler 使用的模型
- `summarizer_model_id`: Summarizer 使用的模型
- `agent_config.model_id`: Agent 使用的模型

### 工具配置

#### 网络搜索工具

支持多种搜索引擎：
- Firecrawl（推荐）
- Google
- Bing
- DuckDuckGo
- Baidu

```yaml
web_searcher_tool_config:
  type: web_searcher_tool
  engine: Firecrawl
  num_results: 5
  fetch_content: true
  max_length: 4096
```

#### MCP 工具集成

支持 Model Context Protocol (MCP) 工具：

```yaml
mcp_tools_config:
  mcpServers:
    LocalMCP:
      command: python
      args:
        - src/mcp/server.py
```

### 失败重试机制

系统实现了智能的失败重试机制：

- `subtask_failure_threshold`: 子任务失败率阈值（0.0-1.0）
- `max_failures_per_parent`: 每个父任务的最大失败次数
- 当子任务失败率超过阈值时，父任务的 `failure_count` 会增加
- 当 `failure_count >= max_failures_per_parent` 时，父任务不再被选择

## 📊 输出说明

### 日志文件

执行过程中的日志会保存到 `log_path` 指定的文件（默认：`log.txt`）。

### 结果文件

最终结果会保存到 `save_path` 指定的文件（默认：`dra.jsonl`），格式为 JSONL（每行一个 JSON 对象）。

### 结果格式

```json
{
  "task": "研究任务描述",
  "result": "最终研究结果",
  "steps": null,
  "token_usage": null
}
```

## 🛠️ 开发指南

### 添加新的智能体类型

1. 在 `src/agent/` 下创建新的智能体目录
2. 实现继承自 `BaseGraphAgent` 的智能体类
3. 使用 `@AGENT.register_module` 装饰器注册

```python
from src.agent.common import BaseGraphAgent
from src.registry import AGENT

@AGENT.register_module(name="my_agent", force=True)
class MyAgent(BaseGraphAgent):
    def __init__(self, config, model, tools, **kwargs):
        # 初始化逻辑
        super().__init__(name="my_agent", ...)
    
    def _build_graph(self):
        # 构建 LangGraph 工作流
        pass
```

### 添加新的工具

1. 在 `src/tools/` 下创建工具类
2. 继承 `Tool` 基类并实现 `forward` 方法
3. 使用 `@TOOL.register_module` 装饰器注册

```python
from src.tools.tools import Tool
from src.registry import TOOL

@TOOL.register_module(name="my_tool", force=True)
class MyTool(Tool):
    name = "my_tool"
    description = "工具描述"
    parameters = {...}
    
    async def forward(self, **kwargs):
        # 工具执行逻辑
        pass
```

