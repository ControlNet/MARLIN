from typing import List, Any, TypeVar, Union
from typing import Type, Dict

import numpy as np
import torchvision
import yaml
from einops import rearrange
from numpy import ndarray
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


def crop_with_padding(image: ndarray, x1: int, x2: int, y1: int, y2: int, pad_value: Union[int, float] = 0.,
    batch: bool = False
) -> ndarray:
    assert y2 > y1 and x2 > x1, "Should follow y2 > y1 and x2 > x1"

    if not batch:
        image = image[np.newaxis, ...]

    crop_shape = np.array([y2 - y1, x2 - x1])

    if len(image.shape) == 3:
        b, h, w = image.shape
        cropped = np.full((b, *crop_shape), pad_value, dtype=image.dtype)
    elif len(image.shape) == 4:
        b, h, w, c = image.shape
        cropped = np.full((b, *crop_shape, c), pad_value, dtype=image.dtype)
    else:
        raise ValueError("Invalid shape, the image should be one of following shapes: ([B,] H, W) or ([B,] H, W, C)")

    # compute cropped index of image
    image_y_start, image_x_start = np.clip([y1, x1], 0, [h, w])
    image_y_end, image_x_end = np.clip([y2, x2], 0, [h, w])

    # compute target index of output
    crop_y_start, crop_x_start = np.clip([-y1, -x1], 0, crop_shape)
    crop_y_end, crop_x_end = crop_shape - np.clip([y2 - h, x2 - w], 0, crop_shape)

    # assign values
    cropped[:, crop_y_start:crop_y_end, crop_x_start:crop_x_end] = \
        image[:, image_y_start:image_y_end, image_x_start:image_x_end]

    return cropped if batch else cropped[0]
