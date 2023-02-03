import os
from math import ceil
from typing import Collection, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from util.misc import sample_indexes


class YoutubeFace(Dataset):
    seg_groups = [
        [2, 4],  # right eye
        [3, 5],  # left eye
        [6],  # nose
        [7, 8, 9],  # mouth
        [10],  # hair
        [1],  # skin
        [0]  # background
    ]

    def __init__(self,
        root_dir: str,
        split: str,
        clip_frames: int,  # T = 16
        temporal_sample_rate: int,  # 2
        patch_size: int,  # 16
        tubelet_size: int,  # 2
        mask_percentage_target: float = 0.8,  # 0.9
        mask_strategy: str = "fasking",
        take_num: Optional[int] = None
    ):
        self.img_size = 224
        self.root_dir = root_dir
        self.clip_frames = clip_frames
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.temporal_sample_rate = temporal_sample_rate
        self.mask_percentage_target = 1 - mask_percentage_target
        self.mask_unit_num = self.img_size // self.patch_size
        assert split in ("train", "val")

        self.patch_masking = True

        if mask_strategy == "fasking":  # Masking components first
            self.face_strategy = True
            self.face_masking_opposite = False
        elif mask_strategy == "fasking_opp":  # Masking background and skin first
            self.face_strategy = True
            self.face_masking_opposite = True
        elif mask_strategy in ("random", "tube", "frame"):  # Masking strategy from VideoMAE
            self.face_strategy = False
            self.face_masking_opposite = None
        else:
            raise ValueError("mask_strategy must be one of 'fasking', 'fasking_opp', 'random', 'tube' and 'frame'")

        self.mask_strategy = mask_strategy
        self.metadata = pd.read_csv(os.path.join(root_dir, f"{split}_set.csv"))
        if take_num:
            self.metadata = self.metadata.iloc[:take_num]

    def __getitem__(self, index):
        meta = self.metadata.iloc[index]
        files = sorted(os.listdir(os.path.join(self.root_dir, "crop_images_DB", meta.path)))
        indexes = self._sample_indexes(len(files))
        assert len(indexes) == self.clip_frames

        video = torch.zeros(self.clip_frames, self.img_size, self.img_size, 3, dtype=torch.float32)
        if self.patch_masking:
            masks = torch.zeros(self.clip_frames // self.tubelet_size,
                self.mask_unit_num, self.mask_unit_num, dtype=torch.float32)
        else:
            masks = torch.zeros(self.clip_frames, self.img_size, self.img_size, dtype=torch.float32)

        if self.face_strategy:
            if self.face_masking_opposite:
                keep_queue = np.concatenate([np.random.permutation(5), [5, 6]])
            else:
                keep_queue = np.concatenate([[6, 5], np.random.permutation(5)])
        else:
            keep_queue = None

        for i in range(self.clip_frames):
            img = cv2.imread(os.path.join(self.root_dir, "crop_images_DB", meta.path, files[indexes[i]]))
            video[i] = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255)
            if i % self.tubelet_size == 0:
                mask = self.gen_mask(meta.path, files[indexes[i]].replace(".jpg", ".npy"), keep_queue)
                masks[i // self.tubelet_size] = mask

        if self.mask_strategy == "tube":
            first_mask = masks[0].flatten().bool()
            target_visible_num = ceil(len(first_mask) * self.mask_percentage_target)
            visible_indexes = first_mask.nonzero().flatten()
            extra_indexes = np.random.choice(visible_indexes, len(visible_indexes) - target_visible_num,
                replace=False)
            first_mask[extra_indexes] = False
            masks = first_mask.repeat(self.clip_frames // self.tubelet_size)
        elif self.mask_strategy == "frame":
            frame_mask = torch.ones(self.clip_frames // self.tubelet_size)
            target_visible_num = ceil(len(frame_mask) * self.mask_percentage_target)
            visible_indexes = frame_mask.nonzero().flatten()
            extra_indexes = np.random.choice(visible_indexes, len(visible_indexes) - target_visible_num,
                replace=False)
            frame_mask[extra_indexes] = 0.0
            masks = rearrange(frame_mask, "t -> t 1 1").expand_as(masks).flatten().bool()

        else:
            masks = masks.flatten().bool()

        # normalize the masking to strictly target percentage for batch computation.
        target_visible_num = int(len(masks) * self.mask_percentage_target)
        visible_indexes = masks.nonzero().flatten()
        extra_indexes = np.random.choice(visible_indexes, len(visible_indexes) - target_visible_num,
            replace=False)
        masks[extra_indexes] = False

        return rearrange(video, "t h w c -> c t h w"), masks.flatten().bool()

    def gen_mask(self, dir_path, file_name, keep_queue: Collection[int]) -> Tensor:
        # we follow the tube style masking, where the masking is only determined by the first frame
        # 0 -> masked, 1 -> visible
        patch_masking = torch.zeros(self.mask_unit_num, self.mask_unit_num, dtype=torch.float32)
        # if mask randomly, early return
        if not self.face_strategy:
            patch_masking[:] = 1
            return patch_masking

        # load face parsing results
        npy_file = os.path.join(self.root_dir, "face_parsing_images_DB", dir_path, file_name)
        face_parsing = torch.from_numpy(np.load(npy_file))
        if face_parsing.shape[0] > 0:
            terminate = False
            for i in keep_queue:
                if terminate:
                    break

                for comp_value in self.seg_groups[i]:
                    patch_masking = torch.maximum(patch_masking, F.max_pool2d((face_parsing == comp_value).float(),
                        kernel_size=self.patch_size)[0])
                    if patch_masking.mean() >= self.mask_percentage_target:
                        terminate = True
                        break
        else:
            patch_masking[:] = 1.

        return patch_masking

    def __len__(self) -> int:
        return len(self.metadata)

    def _sample_indexes(self, num_frames: int) -> Tensor:
        return sample_indexes(num_frames, self.clip_frames, self.temporal_sample_rate)


class YoutubeFaceDataModule(LightningDataModule):

    def __init__(self,
        root_dir: str,
        batch_size: int,
        clip_frames: int,
        temporal_sample_rate: int,
        patch_size: int,
        tubelet_size: int,
        mask_percentage_target: float = 0.8,
        mask_strategy: str = "face",
        num_workers: int = 0,
        take_train: Optional[int] = None,
        take_val: Optional[int] = None
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.clip_frames = clip_frames
        self.temporal_sample_rate = temporal_sample_rate
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.mask_percentage_target = mask_percentage_target
        self.mask_strategy = mask_strategy
        self.take_train = take_train
        self.take_val = take_val
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = YoutubeFace(
            root_dir=self.root_dir,
            split="train",
            clip_frames=self.clip_frames,
            temporal_sample_rate=self.temporal_sample_rate,
            patch_size=self.patch_size,
            tubelet_size=self.tubelet_size,
            mask_percentage_target=self.mask_percentage_target,
            mask_strategy=self.mask_strategy,
            take_num=self.take_train
        )

        self.val_dataset = YoutubeFace(
            root_dir=self.root_dir,
            split="val",
            clip_frames=self.clip_frames,
            temporal_sample_rate=self.temporal_sample_rate,
            patch_size=self.patch_size,
            tubelet_size=self.tubelet_size,
            mask_percentage_target=self.mask_percentage_target,
            mask_strategy=self.mask_strategy,
            take_num=self.take_val
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
            pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
            pin_memory=True)
