import shutil
from collections import deque
from pathlib import Path
from typing import Generator
from urllib.request import urlretrieve

import cv2
import ffmpeg
import torch
from torch import Tensor
from torch.nn import Linear, Module

from .decoder import MarlinDecoder
from .encoder import MarlinEncoder
from .util import read_video, padding_video, DownloadProgressBar


class Marlin(Module):
    MARLIN_FULL_URL = "https://github.com/ControlNet/MARLIN/releases/download/model_v1/marlin.full.pt"
    MARLIN_ENCODER_URL = "https://github.com/ControlNet/MARLIN/releases/download/model_v1/marlin.encoder.pt"

    def __init__(self,
        img_size=224,
        patch_size=16,
        n_frames=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=384,
        decoder_depth=4,
        decoder_num_heads=6,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        norm_layer="LayerNorm",
        init_values=0.,
        tubelet_size=2,
        as_feature_extractor=True,
    ):
        super().__init__()
        self.encoder = MarlinEncoder(
            img_size=img_size,
            patch_size=patch_size,
            n_frames=n_frames,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size
        )
        self.as_feature_extractor = as_feature_extractor
        self.clip_frames = n_frames
        if as_feature_extractor:
            self.enc_dec_proj = None
            self.decoder = None
        else:
            self.decoder = MarlinDecoder(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=decoder_embed_dim,
                depth=decoder_depth,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                norm_layer=norm_layer,
                init_values=init_values,
                tubelet_size=tubelet_size
            )

            self.enc_dec_proj = Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        if self.as_feature_extractor:
            raise RuntimeError("For feature extraction, please use `extract_features` or `extract_video`.")
        else:
            assert mask is not None
            x = self.encoder(x, mask)
            x = self.enc_dec_proj(x)
            x = self.decoder(x, mask)
        return x

    @property
    def device(self):
        return self.encoder.norm.weight.device

    @torch.no_grad()
    def extract_features(self, x: Tensor, keep_seq: bool = True):
        """Extract features for one video clip (v)"""
        self.eval()
        return self.encoder.extract_features(x, seq_mean_pool=not keep_seq)

    def _crop_face(self, v: Tensor) -> Tensor:
        raise NotImplementedError("Please crop the face and resized to 224x224 before passing the video to the model.")

    @torch.no_grad()
    def extract_video(self, video_path: str, sample_rate: int = 2, stride: int = 16, reduction: str = "none",
        keep_seq: bool = False
    ) -> Tensor:
        self.eval()
        features = []
        for v in self._load_video(video_path, sample_rate, stride):
            # v: (1, C, T, H, W)
            if v.shape[3:] != (224, 224):
                v = self._crop_face(v)
            assert v.shape[3:] == (224, 224)
            features.append(self.extract_features(v, keep_seq=keep_seq))

        features = torch.cat(features)  # (N, 768)

        if reduction == "mean":
            return features.mean(dim=0)
        elif reduction == "max":
            return features.max(dim=0)[0]

        return features

    def _load_video(self, video_path: str, sample_rate: int, stride: int) -> Generator[Tensor, None, None]:
        probe = ffmpeg.probe(video_path)
        total_frames = int(probe["streams"][0]["nb_frames"])
        if total_frames <= self.clip_frames:
            video = read_video(video_path, channel_first=True) / 255  # (T, C, H, W)
            # pad frames to 16
            v = padding_video(video, self.clip_frames, "same")  # (T, C, H, W)
            assert v.shape[0] == self.clip_frames
            yield v.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
        elif total_frames < self.clip_frames * sample_rate:
            video = read_video(video_path, channel_first=True) / 255  # (T, C, H, W)
            # use first 16 frames
            v = video[:self.clip_frames]
            yield v.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)
        else:
            # extract features based on sliding window
            cap = cv2.VideoCapture(video_path)
            deq = deque(maxlen=self.clip_frames)

            clip_start_indexes = list(range(0, total_frames - self.clip_frames * sample_rate, stride))
            clip_end_indexes = [i + self.clip_frames * sample_rate - 1 for i in clip_start_indexes]

            current_index = -1
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                current_index += 1
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame).permute(2, 0, 1) / 255  # (C, H, W)

                for _ in range(sample_rate - 1):
                    cap.read()
                    current_index += 1

                deq.append(frame)
                if current_index in clip_end_indexes:
                    v = torch.stack(list(deq))  # (T, C, H, W)
                    yield v.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)

            cap.release()

    @classmethod
    def from_file(cls, pt_path: str) -> "Marlin":
        state_dict = torch.load(pt_path, map_location="cpu")
        # determine if the checkpoint is full model or encoder only.
        for key in state_dict.keys():
            if key.startswith("decoder."):
                as_feature_extractor = False
                break
        else:
            as_feature_extractor = True
        model = cls(as_feature_extractor=as_feature_extractor)
        model.load_state_dict(state_dict)
        return model

    @classmethod
    def from_online(cls, full_model: bool = False) -> "Marlin":
        url = cls.MARLIN_FULL_URL if full_model else cls.MARLIN_ENCODER_URL
        path = Path(".marlin")
        path.mkdir(exist_ok=True)
        file = path / url.split("/")[-1]
        if not file.exists():
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="Downloading Marlin model") as pb:
                urlretrieve(url, filename=file, reporthook=pb.update_to)
        return cls.from_file(str(file))

    @classmethod
    def clean_cache(cls, verbose: bool = True) -> None:
        path = Path(".marlin")
        if path.exists():
            shutil.rmtree(path)
            if verbose:
                print("Marlin checkpoints cache cleaned.")
