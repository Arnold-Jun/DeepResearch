from langchain_openai import ChatOpenAI

from src.models.base import Model


def to_langchain_model(model: Model) -> ChatOpenAI:
    """将 Model 实例转换为 LangChain 兼容的 ChatOpenAI 模型。"""
    if not hasattr(model, 'model_id') or not hasattr(model, 'api_base') or not hasattr(model, 'api_key'):
        raise ValueError(
            f"模型 {type(model).__name__} 不支持转换为 LangChain 模型。"
            "需要具有 model_id、api_base 和 api_key 属性。"
        )
    
    return ChatOpenAI(
        model=model.model_id,
        base_url=model.api_base,
        api_key=model.api_key,
        temperature=0,
        extra_body={"enable_thinking": False},
    )

