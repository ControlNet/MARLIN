from transformers import AutoModel, AutoConfig

from marlin_huggingface.config import MarlinConfig
from marlin_huggingface.marlin import Marlin, MarlinModel

MarlinConfig.register_for_auto_class()
MarlinModel.register_for_auto_class("AutoModel")
AutoConfig.register("marlin", MarlinConfig)
AutoModel.register(MarlinConfig, MarlinModel)

__all__ = ["Marlin", "MarlinModel", "MarlinConfig"]
