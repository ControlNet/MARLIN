from typing import Optional

import torch
from torch import Tensor
from torch.nn import Linear, Module
from transformers import PreTrainedModel

from .encoder import MarlinEncoder
from .decoder import MarlinDecoder

from .config import MarlinConfig


class Marlin(Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        n_frames: int,
        encoder_embed_dim: int,
        encoder_depth: int,
        encoder_num_heads: int,
        decoder_embed_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_scale: Optional[float],
        drop_rate: float,
        attn_drop_rate: float,
        norm_layer: str,
        init_values: float,
        tubelet_size: int,
        as_feature_extractor: bool = True,
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
            tubelet_size=tubelet_size,
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
                tubelet_size=tubelet_size,
            )

            self.enc_dec_proj = Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        if self.as_feature_extractor:
            raise RuntimeError(
                "For feature extraction, please use `extract_features` or `extract_video`."
            )
        else:
            assert mask is not None
            x = self.encoder(x, mask)
            x = self.enc_dec_proj(x)
            x = self.decoder(x, mask)
        return x

    @property
    def device(self):
        return self.encoder.norm.weight.device

    def extract_features(self, x: Tensor, keep_seq: bool = True):
        """Extract features for one video clip (v)"""
        if self.training:
            return self.encoder.extract_features(x, seq_mean_pool=not keep_seq)
        else:
            with torch.no_grad():
                return self.encoder.extract_features(x, seq_mean_pool=not keep_seq)


class MarlinModel(PreTrainedModel):
    config_class = MarlinConfig

    def __init__(self, config: MarlinConfig):
        super().__init__(config)
        self.config = config
        self.marlin = Marlin(
            img_size=config.img_size,
            patch_size=config.patch_size,
            n_frames=config.n_frames,
            encoder_embed_dim=config.encoder_embed_dim,
            encoder_depth=config.encoder_depth,
            encoder_num_heads=config.encoder_num_heads,
            decoder_embed_dim=config.decoder_embed_dim,
            decoder_depth=config.decoder_depth,
            decoder_num_heads=config.decoder_num_heads,
            mlp_ratio=config.mlp_ratio,
            qkv_bias=config.qkv_bias,
            qk_scale=config.qk_scale,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            norm_layer=config.norm_layer,
            init_values=config.init_values,
            tubelet_size=config.tubelet_size,
        )

    def forward(self, x: Tensor, keep_seq: bool = True):
        return self.marlin.extract_features(x, keep_seq=keep_seq)
