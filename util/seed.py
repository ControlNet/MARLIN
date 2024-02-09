import random
from typing import Callable

import numpy as np
import torch
from torch import Generator


class Seed:
    seed: int = None

    @classmethod
    def torch(cls, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @classmethod
    def python(cls, seed: int) -> None:
        random.seed(seed)

    @classmethod
    def numpy(cls, seed: int) -> None:
        np.random.seed(seed)

    @classmethod
    def set(cls, seed: int, use_deterministic_algorithms: bool = False) -> None:
        cls.torch(seed)
        cls.python(seed)
        cls.numpy(seed)
        cls.seed = seed
        torch.use_deterministic_algorithms(use_deterministic_algorithms)

    @classmethod
    def _is_set(cls) -> bool:
        return cls.seed is not None

    @classmethod
    def get_loader_worker_init(cls) -> Callable[[int], None]:
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        if cls._is_set():
            return seed_worker
        else:
            return lambda x: None

    @classmethod
    def get_torch_generator(cls, device="cpu") -> Generator:
        g = torch.Generator(device)
        g.manual_seed(cls.seed)
        return g
