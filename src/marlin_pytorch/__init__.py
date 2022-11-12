import os.path

from .model import Marlin

__all__ = [
    "Marlin",
]

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file:
    __version__ = file.read()
