"""
Custom Registry implementation to replace mmengine.registry.Registry
"""
from typing import Dict, Type, Any, Optional, List
import importlib
import inspect
from pathlib import Path


class Registry:
    """
    A registry for mapping strings to classes or functions.
    
    Args:
        name (str): Registry name.
        locations (list, optional): Locations to research for modules. Defaults to None.
    """
    
    def __init__(self, name: str, locations: Optional[List[str]] = None):
        self.name = name
        self.locations = locations or []
        self._modules: Dict[str, Type] = {}
        self._imported = False
    
    def _import_modules(self):
        """Import modules from specified locations."""
        if self._imported:
            return
        
        for location in self.locations:
            try:
                # Convert location to module path
                module_path = location.replace('/', '.').replace('\\', '.')
                if module_path.endswith('.py'):
                    module_path = module_path[:-3]
                
                # Import the module
                importlib.import_module(module_path)
            except Exception as e:
                # Silently ignore import errors for optional modules
                pass
        
        self._imported = True
    
    def register_module(self, name: str = None, force: bool = False):
        """
        Register a module.
        
        Args:
            name (str, optional): Module name. If None, use class name.
            force (bool): Whether to override existing module. Defaults to False.
        
        Returns:
            callable: Decorator function.
        """
        def decorator(cls):
            module_name = name if name is not None else cls.__name__
            
            if module_name in self._modules and not force:
                raise KeyError(
                    f"{module_name} is already registered in {self.name}. "
                    f"Use force=True to override."
                )
            
            self._modules[module_name] = cls
            return cls
        
        return decorator
    
    def get(self, name: str) -> Type:
        """
        Get a registered module.
        
        Args:
            name (str): Module name.
        
        Returns:
            Type: Registered class or function.
        """
        self._import_modules()
        
        if name not in self._modules:
            raise KeyError(
                f"{name} is not registered in {self.name}. "
                f"Available modules: {list(self._modules.keys())}"
            )
        
        return self._modules[name]
    
    def build(self, config: Dict[str, Any], **kwargs) -> Any:
        """
        Build an instance from config.
        
        Args:
            config (dict): Config dict. Must contain 'type' key.
            **kwargs: Additional arguments to pass to constructor.
        
        Returns:
            object: Instance of the registered class.
        """
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, got {type(config)}")
        
        if 'type' not in config:
            raise KeyError(f"config must contain 'type' key, got {config}")
        
        module_type = config['type']
        cls = self.get(module_type)
        
        # Extract constructor arguments from config
        # Exclude 'type' and merge with kwargs
        constructor_args = {k: v for k, v in config.items() if k != 'type'}
        constructor_args.update(kwargs)
        
        # Get constructor signature to filter valid arguments
        try:
            sig = inspect.signature(cls.__init__)
            valid_params = set(sig.parameters.keys()) - {'self'}
            filtered_args = {k: v for k, v in constructor_args.items() if k in valid_params}
        except (ValueError, TypeError):
            # If signature inspection fails, use all args
            filtered_args = constructor_args
        
        return cls(**filtered_args)
    
    def __contains__(self, name: str) -> bool:
        """Check if a module is registered."""
        self._import_modules()
        return name in self._modules
    
    def __getitem__(self, name: str) -> Type:
        """Get a registered module using [] syntax."""
        return self.get(name)
    
    def __repr__(self) -> str:
        self._import_modules()
        return f"Registry(name={self.name}, modules={list(self._modules.keys())})"


# Create registry instances
DATASET = Registry('dataset', locations=['src.dataset'])
TOOL = Registry('tool', locations=['src.tools'])
AGENT = Registry(
    'agent',
    locations=[
        # Import concrete agent modules so @AGENT.register_module decorators run.
        'src.agent.deep_researcher_agent.agent',
        'src.agent.deep_analyzer_agent.agent',
        'src.agent.browser_use_agent.agent',
    ],
)
