import os
import unittest
from typing import Optional

from .test_marlin_pytorch import TestMarlinPytorch


class MarlinViTBase(unittest.TestCase, TestMarlinPytorch):
    MODEL_NAME: Optional[str] = "marlin_vit_base_ytf"
    MODEL_ENCODER_PATH: Optional[str] = os.path.join("test", "model", f"marlin_vit_base_ytf.encoder.pt")
    MODEL_FULL_PATH: Optional[str] = os.path.join("test", "model", "marlin_vit_base_ytf.full.pt")
    EMBEDDING_SIZE: Optional[int] = 768
