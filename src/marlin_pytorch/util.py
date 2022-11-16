from typing import List, Any, TypeVar
from typing import Type, Dict

import torchvision
import yaml
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F
from tqdm.auto import tqdm


def read_video(path: str, channel_first: bool = True):
    video, audio, info = torchvision.io.read_video(path)
    if channel_first:
        video = rearrange(video, 'T H W C -> T C H W')
    return video


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as file:
        return yaml.load(file, Loader=yaml.Loader)


def padding_video(tensor: Tensor, target: int, padding_method: str = "zero", padding_position: str = "tail") -> Tensor:
    t, c, h, w = tensor.shape
    padding_size = target - t

    pad = _get_padding_pair(padding_size, padding_position)

    if padding_method == "zero":
        return F.pad(tensor, pad=[0, 0, 0, 0, 0, 0] + pad)
    elif padding_method == "same":
        tensor = rearrange(tensor, "t c h w -> c h w t")
        tensor = F.pad(tensor, pad=pad + [0, 0], mode="replicate")
        return rearrange(tensor, "c h w t -> t c h w")
    else:
        raise ValueError("Wrong padding method. It should be zero or tail or average.")


def _get_padding_pair(padding_size: int, padding_position: str) -> List[int]:
    if padding_position == "tail":
        pad = [0, padding_size]
    elif padding_position == "head":
        pad = [padding_size, 0]
    elif padding_position == "average":
        padding_head = padding_size // 2
        padding_tail = padding_size - padding_head
        pad = [padding_head, padding_tail]
    else:
        raise ValueError("Wrong padding position. It should be zero or tail or average.")
    return pad


class DownloadProgressBar(tqdm):
    total: int

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


T = TypeVar("T")


class Singleton:
    all_instances: Dict[Type, object] = {}

    def __new__(cls, clazz: Type[T]) -> T:
        cls.all_instances[clazz] = clazz()
        return cls.all_instances[clazz]
