import os
import subprocess
import atexit
import signal
import threading
import socket

from browser_use import Agent

from src.tools import AsyncTool, ToolResult
from src.tools.browser import Controller
from src.utils import assemble_project_path
from src.registry import TOOL
from src.models import model_manager, to_langchain_model

@TOOL.register_module(name="auto_browser_use_tool", force=True)
class AutoBrowserUseTool(AsyncTool):
    name = "auto_browser_use_tool"
    description = "一个强大的浏览器自动化工具，允许通过各种操作与网页交互。根据给定任务自动浏览网络并提取信息。"
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "要执行的任务"
            },
        },
        "required": ["task"],
    }
    output_type = "any"

    _server_proc = None
    _server_lock = threading.Lock()
    _server_initialized = False

    def __init__(self,
                 model_id: str = "qwen3-8b",
                 ):

        super().__init__()

        self.model_id = model_id
        self.http_server_path = assemble_project_path("src/tools/browser/http_server")
        self.http_save_path = assemble_project_path("src/tools/browser/http_server/local")
        os.makedirs(self.http_save_path, exist_ok=True)

        self._ensure_pdf_server()

    @classmethod
    def _is_port_in_use(cls, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return False
            except OSError:
                return True

    @classmethod
    def _ensure_pdf_server(cls):
        if cls._server_initialized:
            return
        
        with cls._server_lock:
            if cls._server_initialized:
                return
            
            http_server_path = assemble_project_path("src/tools/browser/http_server")
            port = 8080
            
            if cls._is_port_in_use(port):
                cls._server_initialized = True
                return
            
            try:
                cls._server_proc = subprocess.Popen(
                    ["python3", "-m", "http.server", str(port)],
                    cwd=http_server_path,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=None
                )

                @atexit.register
                def shutdown_server():
                    if cls._server_proc:
                        try:
                            cls._server_proc.send_signal(signal.SIGTERM)
                            cls._server_proc.wait(timeout=5)
                        except Exception:
                            cls._server_proc.kill()
                
                cls._server_initialized = True
            except Exception:
                pass

    async def _browser_task(self, task):
        controller = Controller(http_save_path=self.http_save_path)

        assert self.model_id in ['qwen3-8b', 'qwen3-14b', 'qwen3-32b'], f"模型应该在 [qwen3-8b, qwen3-14b, qwen3-32b] 中，但得到 {self.model_id}。请检查您的配置文件。"

        model = model_manager.registered_models[self.model_id]
        
        original_openai_key = os.environ.get("OPENAI_API_KEY")
        browser_agent = None
        try:
            if model.api_key:
                os.environ["OPENAI_API_KEY"] = model.api_key
            
            langchain_model = to_langchain_model(model)

            browser_agent = Agent(
                task=task,
                llm=langchain_model,
                enable_memory=False,
                controller=controller,
                page_extraction_llm=langchain_model,
            )

            history = await browser_agent.run(max_steps=50)
            result_parts = []
            
            extracted_contents = history.extracted_content()
            if extracted_contents:
                result_parts.append("=== 提取的内容 ===\n" + "\n\n".join(extracted_contents))
            
            if hasattr(history, 'messages') and history.messages:
                messages_text = []
                for msg in history.messages:
                    if hasattr(msg, 'content') and msg.content:
                        content = str(msg.content)
                        if len(content) > 500:
                            messages_text.append(f"[{getattr(msg, 'role', 'unknown')}]: {content[:2000]}...")
                if messages_text:
                    result_parts.append("=== 关键消息 ===\n" + "\n\n".join(messages_text))
            
            if hasattr(history, 'steps') and history.steps:
                steps_summary = []
                for i, step in enumerate(history.steps[-10:], 1):
                    step_str = str(step)
                    if len(step_str) > 100:
                        steps_summary.append(f"步骤 {i}: {step_str[:500]}...")
                if steps_summary:
                    result_parts.append("=== 执行步骤摘要 ===\n" + "\n".join(steps_summary))
            
            if not result_parts:
                history_str = str(history)
                if history_str and len(history_str) > 50:
                    result_parts.append(f"=== 浏览历史 ===\n{history_str}")
            
            if not result_parts:
                result_parts.append("任务已完成，但未提取到详细内容。")
            
            return "\n\n".join(result_parts)
        finally:
            if browser_agent and hasattr(browser_agent, 'cleanup'):
                try:
                    await browser_agent.cleanup()
                except Exception:
                    pass
            
            if original_openai_key is not None:
                os.environ["OPENAI_API_KEY"] = original_openai_key
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

    async def forward(self, task: str) -> ToolResult:
        result = await self._browser_task(task)
        return ToolResult(output=result, error=None)