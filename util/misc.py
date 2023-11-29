from __future__ import annotations

import json
import re

import torch
from torch import Tensor


def rename_key(dictionary, old_key, new_key):
    dictionary[new_key] = dictionary.pop(old_key)


def load_official_pretrain_model(model, pth_path):
    state_dict = torch.load(pth_path)["model"]

    mutation_keys = [
        ("mask_token", "decoder.mask_token"),
    ]

    for old_key in state_dict:
        m = re.match(r"(en|de)coder\.blocks\.(\d+?)\.mlp\.fc([1-2])\.(weight|bias)", old_key)
        if m:
            new_key = f"{m[1]}coder.blocks.{m[2]}.mlp.layers.{int(m[3]) - 1}.linear.{m[4]}"
            mutation_keys.append((old_key, new_key))

        if old_key.startswith("encoder_to_decoder"):
            new_key = old_key.replace("encoder_to_decoder", "enc_dec_proj")
            mutation_keys.append((old_key, new_key))

        if old_key.startswith("encoder.patch_embed"):
            new_key = old_key.replace("patch_embed.proj", "patch_embedding.projection")
            mutation_keys.append((old_key, new_key))

    for old_key, new_key in mutation_keys:
        rename_key(state_dict, old_key, new_key)

    return model.load_state_dict(state_dict, strict=False)


def sample_indexes(total_frames: int, n_frames: int, temporal_sample_rate: int) -> Tensor:
    try:
        start_ind = torch.randint(0, total_frames - (n_frames * temporal_sample_rate) + 1, ())
    except RuntimeError as e:
        print(f"total_frames: {total_frames}, n_frames: {n_frames}, temporal_sample_rate: {temporal_sample_rate}")
        raise e
    return torch.arange(n_frames) * temporal_sample_rate + start_ind


def read_text(path: str, encoding: str = "UTF-8") -> str:
    with open(path, "r", encoding=encoding) as file:
        text = file.read()
    return text


def read_json(path: str):
    with open(path, "r") as file:
        return json.load(file)
