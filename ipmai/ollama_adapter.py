"""
Ollama Runtime Adapter for SGLang
创建一个兼容 SGLang Runtime 接口的适配器，内部使用 OpenAI 客户端
"""
from typing import Any, Optional
from openai import OpenAI


class OllamaRuntimeAdapter:
    """适配 SGLang Runtime 接口的 Ollama 适配器"""
    
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.endpoint = None
        self.support_concate_and_append = True
        
    def shutdown(self):
        """关闭连接"""
        pass
    
    def __getattr__(self, name: str) -> Any:
        """委托所有属性访问"""
        # SGLang 可能会访问很多属性，我们提供一个基本的兼容层
        if name in ["endpoint", "support_concate_and_append"]:
            return getattr(self, name)
        # 对于其他属性，返回 None 或抛出 AttributeError
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

