import unittest

import numpy as np
import torch.cuda

from marlin_pytorch import Marlin
import os


class TestMarlinPytorch(unittest.TestCase):
    CROP_VIDEOS = [f"cropped{str(i).zfill(2)}" for i in range(1, 6)]
    WILD_VIDEOS = [f"video{str(i).zfill(2)}" for i in range(1, 6)]
    USE_GPU = torch.cuda.is_available()

    VIT_BASE_NAME = "marlin_vit_base_ytf"
    VIT_LARGE_NAME = "marlin_vit_large_ytf"
    VIT_SMALL_NAME = "marlin_vit_small_ytf"

    VIT_BASE_MODEL_FULL_PATH = os.path.join("test", "model", "marlin_vit_base_ytf.full.pt")
    VIT_BASE_MODEL_ENCODER_PATH = os.path.join("test", "model", "marlin_vit_base_ytf.encoder.pt")
    VIT_LARGE_MODEL_FULL_PATH = os.path.join("test", "model", "marlin_vit_large_ytf.full.pt")
    VIT_LARGE_MODEL_ENCODER_PATH = os.path.join("test", "model", "marlin_vit_large_ytf.encoder.pt")
    VIT_SMALL_MODEL_FULL_PATH = os.path.join("test", "model", "marlin_vit_small_ytf.full.pt")
    VIT_SMALL_MODEL_ENCODER_PATH = os.path.join("test", "model", "marlin_vit_small_ytf.encoder.pt")

    def test_load_full_model_from_file(self):
        Marlin.from_file(self.VIT_BASE_NAME, self.VIT_BASE_MODEL_FULL_PATH)
        self.assertTrue(True)
        Marlin.from_file(self.VIT_LARGE_NAME, self.VIT_LARGE_MODEL_FULL_PATH)
        self.assertTrue(True)
        Marlin.from_file(self.VIT_SMALL_NAME, self.VIT_SMALL_MODEL_FULL_PATH)
        self.assertTrue(True)

    def test_load_encoder_from_file(self):
        Marlin.from_file(self.VIT_BASE_NAME, self.VIT_BASE_MODEL_ENCODER_PATH)
        self.assertTrue(True)
        Marlin.from_file(self.VIT_LARGE_NAME, self.VIT_LARGE_MODEL_ENCODER_PATH)
        self.assertTrue(True)
        Marlin.from_file(self.VIT_SMALL_NAME, self.VIT_SMALL_MODEL_ENCODER_PATH)
        self.assertTrue(True)

    def test_load_full_model_from_online(self):
        Marlin.from_online(self.VIT_BASE_NAME, full_model=True)
        self.assertTrue(True)
        Marlin.from_online(self.VIT_LARGE_NAME, full_model=True)
        self.assertTrue(True)
        Marlin.from_online(self.VIT_SMALL_NAME, full_model=True)
        self.assertTrue(True)

    def test_load_encoder_from_online(self):
        Marlin.from_online(self.VIT_BASE_NAME, full_model=False)
        self.assertTrue(True)
        Marlin.from_online(self.VIT_LARGE_NAME, full_model=False)
        self.assertTrue(True)
        Marlin.from_online(self.VIT_SMALL_NAME, full_model=False)
        self.assertTrue(True)

    def test_extract_wild_video(self):
        model = Marlin.from_file(self.VIT_BASE_NAME, self.VIT_BASE_MODEL_ENCODER_PATH)
        if self.USE_GPU:
            model.cuda()

        for video in self.WILD_VIDEOS:
            feat = model.extract_video(os.path.join("test", "input_sample", f"{video}.mp4"), crop_face=True)
            feat = feat.cpu().numpy()
            true = np.load(os.path.join("test", "output_sample", f"{video}.npy"))
            diff = np.abs(feat - true).mean()
            self.assertTrue(diff < 1.5e-4)

    def test_extract_cropped_video(self):
        model = Marlin.from_file(self.VIT_BASE_NAME, self.VIT_BASE_MODEL_ENCODER_PATH)
        if self.USE_GPU:
            model.cuda()

        for video in self.CROP_VIDEOS:
            feat = model.extract_video(os.path.join("test", "input_sample", f"{video}.mp4"))
            feat = feat.cpu().numpy()
            true = np.load(os.path.join("test", "output_sample", f"{video}.npy"))
            diff = np.abs(feat - true).mean()
            self.assertTrue(diff < 1.5e-4)

    def test_extract_cropped_clip(self):
        model = Marlin.from_file(self.VIT_BASE_NAME, self.VIT_BASE_MODEL_ENCODER_PATH)
        if self.USE_GPU:
            model.cuda()

        x = torch.rand(1, 3, 16, 224, 224).to(model.device)
        self.assertTrue(model.extract_features(x).shape == (1, 1568, 768))
        self.assertTrue(model.extract_features(x, keep_seq=False).shape == (1, 768))

    def test_reconstruct_clip(self):
        model = Marlin.from_file(self.VIT_BASE_NAME, self.VIT_BASE_MODEL_FULL_PATH)
        if self.USE_GPU:
            model.cuda()

        mask = torch.zeros((1, 1568)).to(model.device).bool()
        mask[:, :392] = True
        x = torch.rand(1, 3, 16, 224, 224).to(model.device)
        pred = model(x, mask)
        self.assertTrue(pred.shape == (1, 1176, 1536))

    def test_vit_large_full_model(self):
        model = Marlin.from_online(self.VIT_LARGE_NAME, full_model=True)
        if self.USE_GPU:
            model.cuda()

        x = torch.rand(1, 3, 16, 224, 224).to(model.device)
        self.assertTrue(model.extract_features(x).shape == (1, 1568, 1024))
        self.assertTrue(model.extract_features(x, keep_seq=False).shape == (1, 1024))

        mask = torch.zeros((1, 1568)).to(model.device).bool()
        mask[:, :392] = True
        pred = model(x, mask)
        self.assertTrue(pred.shape == (1, 1176, 1536))

    def test_vit_large_encoder(self):
        model = Marlin.from_online(self.VIT_LARGE_NAME, full_model=False)
        if self.USE_GPU:
            model.cuda()

        x = torch.rand(1, 3, 16, 224, 224).to(model.device)
        self.assertTrue(model.extract_features(x).shape == (1, 1568, 1024))
        self.assertTrue(model.extract_features(x, keep_seq=False).shape == (1, 1024))

    def test_vit_small_full_model(self):
        model = Marlin.from_online(self.VIT_SMALL_NAME, full_model=True)
        if self.USE_GPU:
            model.cuda()

        x = torch.rand(1, 3, 16, 224, 224).to(model.device)
        self.assertTrue(model.extract_features(x).shape == (1, 1568, 384))
        self.assertTrue(model.extract_features(x, keep_seq=False).shape == (1, 384))

        mask = torch.zeros((1, 1568)).to(model.device).bool()
        mask[:, :392] = True
        pred = model(x, mask)
        self.assertTrue(pred.shape == (1, 1176, 1536))

    def test_vit_small_encoder(self):
        model = Marlin.from_online(self.VIT_SMALL_NAME, full_model=False)
        if self.USE_GPU:
            model.cuda()

        x = torch.rand(1, 3, 16, 224, 224).to(model.device)
        self.assertTrue(model.extract_features(x).shape == (1, 1568, 384))
        self.assertTrue(model.extract_features(x, keep_seq=False).shape == (1, 384))
