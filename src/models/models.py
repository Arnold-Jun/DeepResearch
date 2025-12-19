import os
from typing import Any
from openai import AsyncOpenAI

from dotenv import load_dotenv
load_dotenv(verbose=True)

from src.logger import logger
from src.models.openaillm import OpenAIServerModel
from src.utils import Singleton

custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}


class ModelManager(metaclass=Singleton):
    def __init__(self):
        self.registered_models: dict[str, Any] = {}
        
    def init_models(self):
        self._register_qwen_models()

    def _register_qwen_models(self):
        logger.info("Registering Qwen API models")
        
        api_key = os.getenv("DASHSCOPE_API_KEY", None)
        if api_key is None:
            api_key = os.getenv("QWEN_API_KEY", None)
        if api_key is None:
            raise ValueError(
                "未找到 Qwen API key。请设置环境变量 DASHSCOPE_API_KEY 或 QWEN_API_KEY。"
                "例如：export DASHSCOPE_API_KEY='your-api-key-here'"
            )
        logger.info("Using Qwen API key from environment variable")
        
        default_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_base = os.getenv("QWEN_API_BASE", default_api_base)
        if api_base != default_api_base:
            logger.info(f"Using Qwen API base from environment variable: {api_base}")
        else:
            logger.info(f"Using default Qwen API base: {default_api_base}")
        
        # 注册三个 Qwen 模型
        models = [
            {
                "model_name": "qwen3-32b",
                "model_id": "qwen3-32b",
            },
            {
                "model_name": "qwen3-14b",
                "model_id": "qwen3-14b",
            },
            {
                "model_name": "qwen3-8b",
                "model_id": "qwen3-8b",
            },
        ]
        
        for model in models:
            model_name = model["model_name"]
            model_id = model["model_id"]
            
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=api_base,
            )
            model_instance = OpenAIServerModel(
                model_id=model_id,
                http_client=client,
                api_key=api_key,
                api_base=api_base,
                custom_role_conversions=custom_role_conversions,
            )
            # 注册模型
            self.registered_models[model_name] = model_instance
            logger.info(f"Registered Qwen model: {model_name} (model_id: {model_id})")
    
    def get_model(self, model_id: str) -> Any:
        """
        获取已注册的模型
        
        Args:
            model_id: 模型 ID
            
        Returns:
            模型实例
            
        Raises:
            ValueError: 如果模型未注册
        """
        if model_id not in self.registered_models:
            available_models = ', '.join(self.registered_models.keys())
            raise ValueError(
                f"模型 '{model_id}' 未注册。"
                f"可用模型: {available_models}"
            )
        return self.registered_models[model_id]