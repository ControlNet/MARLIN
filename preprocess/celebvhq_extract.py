import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from marlin_pytorch import Marlin
from marlin_pytorch.config import resolve_config

sys.path.append(".")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CelebV-HQ Feature Extraction")
    parser.add_argument("--backbone", type=str)
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()

    model = Marlin.from_online(args.backbone)
    config = resolve_config(args.backbone)
    feat_dir = args.backbone

    model.cuda()
    model.eval()

    raw_video_path = os.path.join(args.data_dir, "cropped")
    all_videos = sorted(list(filter(lambda x: x.endswith(".mp4"), os.listdir(raw_video_path))))
    Path(os.path.join(args.data_dir, feat_dir)).mkdir(parents=True, exist_ok=True)
    for video_name in tqdm(all_videos):
        video_path = os.path.join(raw_video_path, video_name)
        save_path = os.path.join(args.data_dir, feat_dir, video_name.replace(".mp4", ".npy"))
        try:
            feat = model.extract_video(
                video_path, crop_face=False,
                sample_rate=config.tubelet_size, stride=config.n_frames,
                keep_seq=False, reduction="none")

        except Exception as e:
            print(f"Video {video_path} error.", e)
            feat = torch.zeros(0, model.encoder.embed_dim, dtype=torch.float32)
        np.save(save_path, feat.cpu().numpy())
