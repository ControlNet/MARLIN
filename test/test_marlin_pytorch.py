import os
from typing import Optional, Callable

import numpy as np
import torch.cuda

from marlin_pytorch import Marlin


class TestMarlinPytorch:
    assertTrue: Callable
    CROP_VIDEOS = [f"cropped{str(i).zfill(2)}" for i in range(1, 6)]
    WILD_VIDEOS = [f"video{str(i).zfill(2)}" for i in range(1, 6)]
    USE_GPU = torch.cuda.is_available()

    MODEL_NAME: Optional[str] = None
    MODEL_ENCODER_PATH: Optional[str] = None
    MODEL_FULL_PATH: Optional[str] = None
    EMBEDDING_SIZE: Optional[int] = None

    def test_load_full_model_from_file(self):
        Marlin.from_file(self.MODEL_NAME, self.MODEL_FULL_PATH)
        self.assertTrue(True)

    def test_load_encoder_from_file(self):
        Marlin.from_file(self.MODEL_NAME, self.MODEL_ENCODER_PATH)
        self.assertTrue(True)

    def test_load_full_model_from_online(self):
        Marlin.from_online(self.MODEL_NAME, full_model=True)
        self.assertTrue(True)

    def test_load_encoder_from_online(self):
        Marlin.from_online(self.MODEL_NAME, full_model=False)
        self.assertTrue(True)

    def test_extract_wild_video(self):
        if not os.path.exists(os.path.join("test", "output_sample", self.MODEL_NAME)):
            return

        model = Marlin.from_file(self.MODEL_NAME, self.MODEL_ENCODER_PATH)
        if self.USE_GPU:
            model.cuda()

        for video in self.WILD_VIDEOS:
            feat = model.extract_video(os.path.join("test", "input_sample", f"{video}.mp4"), crop_face=True)
            feat = feat.cpu().numpy()
            true = np.load(os.path.join("test", "output_sample", self.MODEL_NAME, f"{video}.npy"))
            diff = np.abs(feat - true).mean()
            self.assertTrue(diff < 1.5e-4)

    def test_extract_cropped_video(self):
        if not os.path.exists(os.path.join("test", "output_sample", self.MODEL_NAME)):
            return

        model = Marlin.from_file(self.MODEL_NAME, self.MODEL_ENCODER_PATH)
        if self.USE_GPU:
            model.cuda()

        for video in self.CROP_VIDEOS:
            feat = model.extract_video(os.path.join("test", "input_sample", f"{video}.mp4"))
            feat = feat.cpu().numpy()
            true = np.load(os.path.join("test", "output_sample", self.MODEL_NAME, f"{video}.npy"))
            diff = np.abs(feat - true).mean()
            self.assertTrue(diff < 1.5e-4)

    def test_extract_cropped_clip(self):
        model = Marlin.from_file(self.MODEL_NAME, self.MODEL_ENCODER_PATH)
        if self.USE_GPU:
            model.cuda()

        x = torch.rand(1, 3, 16, 224, 224).to(model.device)
        self.assertTrue(model.extract_features(x).shape == (1, 1568, self.EMBEDDING_SIZE))
        self.assertTrue(model.extract_features(x, keep_seq=False).shape == (1, self.EMBEDDING_SIZE))

    def test_reconstruct_clip(self):
        model = Marlin.from_file(self.MODEL_NAME, self.MODEL_FULL_PATH)
        if self.USE_GPU:
            model.cuda()

        mask = torch.zeros((1, 1568)).to(model.device).bool()
        mask[:, :392] = True
        x = torch.rand(1, 3, 16, 224, 224).to(model.device)
        pred = model(x, mask)
        self.assertTrue(pred.shape == (1, 1176, 1536))
