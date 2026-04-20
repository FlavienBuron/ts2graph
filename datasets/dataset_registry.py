from typing import Dict, Type

from datasets.dataloaders.graphloader import GraphLoader


class DatasetRegistry:
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str, variant: str = "default"):
        """
        Decorator to register a dataset loader class
        """

        def wrapper(target_class: Type):
            config_key = f"{name}_{variant}" if variant != "default" else name

            cls._registry[config_key] = target_class
            return target_class

        return wrapper

    @classmethod
    def get(cls, config: Dict) -> GraphLoader:
        """Get loader class by config key"""
        config_key = config["name"]
        if config_key not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown dataset config: '{config_key}'.Available: {available}"
            )
        loader_class = cls._registry[config_key]
        return loader_class(**config)
