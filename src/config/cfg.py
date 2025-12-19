"""
简单干净的配置实现。
支持从 YAML 文件加载，支持 _base_ 继承。
"""
import os
import yaml
from argparse import Namespace
from typing import Any, Dict

from dotenv import load_dotenv
load_dotenv(verbose=True)

from src.utils import assemble_project_path, Singleton
from src.logger import logger


class Config:
    """
    从 YAML 文件加载的简单配置类。
    支持 _base_ 继承、字典和属性访问。
    """
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        self._config_dict = config_dict or {}
    
    @classmethod
    def fromfile(cls, filename: str) -> 'Config':
        """从 YAML 文件加载配置。"""
        filename = assemble_project_path(filename)
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Config file not found: {filename}")
        
        with open(filename, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f) or {}
        
        config = cls()
        
        # Handle _base_ inheritance
        if '_base_' in yaml_data:
            base_path = os.path.join(os.path.dirname(filename), yaml_data['_base_'])
            if os.path.exists(base_path):
                base_config = cls.fromfile(base_path)
                config._config_dict = base_config._config_dict.copy()
            yaml_data = {k: v for k, v in yaml_data.items() if k != '_base_'}
        
        # Merge current config
        cls._merge_dict(config._config_dict, yaml_data)
        
        return config
    
    @staticmethod
    def _merge_dict(base: Dict, update: Dict) -> None:
        """递归合并字典。"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Config._merge_dict(base[key], value)
            else:
                base[key] = value
    
    def merge_from_dict(self, options: Dict[str, Any]) -> None:
        """将选项合并到配置中。"""
        self._merge_dict(self._config_dict, options)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值。"""
        return self._config_dict.get(key, default)
    
    def __getattr__(self, key: str) -> Any:
        """获取属性。"""
        if key.startswith('_'):
            return super().__getattribute__(key)
        if key in self._config_dict:
            return self._config_dict[key]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    
    def __setattr__(self, key: str, value: Any) -> None:
        """设置属性。"""
        if key.startswith('_') or key == '_config_dict':
            super().__setattr__(key, value)
        else:
            if not hasattr(self, '_config_dict'):
                super().__setattr__('_config_dict', {})
            self._config_dict[key] = value
    
    def __getitem__(self, key: str) -> Any:
        """类似字典的访问。"""
        return self._config_dict[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """类似字典的赋值。"""
        self._config_dict[key] = value
    
    def __contains__(self, key: str) -> bool:
        """检查键是否存在。"""
        return key in self._config_dict
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置返回为字典。"""
        return self._config_dict.copy()
    
    def __repr__(self) -> str:
        return f"Config({list(self._config_dict.keys())})"


def process_general(config: Config) -> Config:
    """处理通用配置设置。"""
    config.exp_path = assemble_project_path(os.path.join(config.workdir, config.tag))
    os.makedirs(config.exp_path, exist_ok=True)
    config.log_path = os.path.join(config.exp_path, getattr(config, 'log_path', 'dra.log'))
    logger.info(f"| Arguments Log file: {config.log_path}")
    if "save_path" in config:
        config.save_path = os.path.join(config.exp_path, getattr(config, 'save_path', 'dra.json'))
    return config


def process_mcp(config: Config) -> Config:
    """处理 MCP 配置。"""
    import sys
    if "mcp_tools_config" in config:
        mcp_servers = config['mcp_tools_config']['mcpServers']
        if 'LocalMCP' in mcp_servers:
            # Use current Python interpreter to ensure correct environment
            mcp_servers['LocalMCP']['command'] = sys.executable
            args = mcp_servers['LocalMCP'].get('args', [])
            args = [assemble_project_path(item) if isinstance(item, str) else item for item in args]
            config['mcp_tools_config']['mcpServers']['LocalMCP']['args'] = args
    return config


class ConfigManager(Config, metaclass=Singleton):
    """单例配置管理器。"""
    
    def __init__(self):
        super().__init__()
    
    def init_config(self, config_path: str, args: Namespace) -> None:
        """从文件和命令行参数初始化配置。"""
        # Load config from YAML file
        mmconfig = Config.fromfile(assemble_project_path(config_path))
        
        # Collect options from args
        cfg_options = getattr(args, 'cfg_options', None) or {}
        for item in args.__dict__:
            if item not in ['config', 'cfg_options'] and args.__dict__[item] is not None:
                cfg_options[item] = args.__dict__[item]
        
        # Merge options
        mmconfig.merge_from_dict(cfg_options)
        
        # Process configurations
        mmconfig = process_general(mmconfig)
        mmconfig = process_mcp(mmconfig)
        
        # Update self
        self._config_dict = mmconfig._config_dict
    
    @property
    def pretty_text(self) -> str:
        """获取格式化的配置文本。"""
        import json
        return json.dumps(self._config_dict, indent=2, ensure_ascii=False)


# 创建单例实例
config = ConfigManager()
