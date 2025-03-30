from transformers import AutoModel, AutoConfig

from .config import MarlinConfig
from .marlin import Marlin, MarlinModel

MarlinConfig.register_for_auto_class()
MarlinModel.register_for_auto_class()
AutoConfig.register("marlin", MarlinConfig)
AutoModel.register(MarlinConfig, MarlinModel)

__all__ = ["Marlin", "MarlinModel", "MarlinConfig"]
