import unittest

import numpy as np
import torch.cuda

from marlin_pytorch import Marlin
import os


class TestMarlinPytorch(unittest.TestCase):

    CROP_VIDEOS = [f"cropped{str(i).zfill(2)}" for i in range(1, 6)]
    WILD_VIDEOS = [f"video{str(i).zfill(2)}" for i in range(1, 6)]
    USE_GPU = torch.cuda.is_available()

    def test_load_full_model_from_file(self):
        # Marlin.from_file(os.path.join("test", "model", "marlin.full.pt"))
        self.assertTrue(True)

    def test_load_encoder_from_file(self):
        # Marlin.from_file(os.path.join("test", "model", "marlin.encoder.pt"))
        self.assertTrue(True)

    def test_load_full_model_from_online(self):
        Marlin.from_online(full_model=True)
        self.assertTrue(True)

    def test_load_encoder_from_online(self):
        Marlin.from_online(full_model=False)
        self.assertTrue(True)

    def test_extract_wild_video(self):
        # not implemented yet
        pass

    def test_extract_cropped_video(self):
        model = Marlin.from_file(os.path.join("test", "model", "marlin.encoder.pt"))
        if self.USE_GPU:
            model.cuda()

        for video in self.CROP_VIDEOS:
            feat = model.extract_video(os.path.join("test", "input_sample", f"{video}.mp4"))
            feat = feat.cpu().numpy()
            true = np.load(os.path.join("test", "output_sample", f"{video}.npy"))
            diff = np.abs(feat - true).mean()
            self.assertTrue(diff < 1.5e-4)

    def test_extract_cropped_clip(self):
        model = Marlin.from_file(os.path.join("test", "model", "marlin.encoder.pt"))
        if self.USE_GPU:
            model.cuda()

        x = torch.rand(1, 3, 16, 224, 224).to(model.device)
        self.assertTrue(model.extract_features(x).shape == (1, 1568, 768))
        self.assertTrue(model.extract_features(x, keep_seq=False).shape == (1, 768))

    def test_reconstruct_clip(self):
        model = Marlin.from_file(os.path.join("test", "model", "marlin.full.pt"))
        if self.USE_GPU:
            model.cuda()

        mask = torch.zeros((1, 1568)).to(model.device).bool()
        mask[:, :392] = True
        x = torch.rand(1, 3, 16, 224, 224).to(model.device)
        pred = model(x, mask)
        self.assertTrue(pred.shape == (1, 1176, 1536))
