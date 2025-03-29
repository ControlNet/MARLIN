from transformers import PretrainedConfig


class MarlinConfig(PretrainedConfig):
    model_type = "marlin"

    def __init__(self, **kwargs):
        self.img_size = kwargs.pop("img_size", None)
        self.patch_size = kwargs.pop("patch_size", None)
        self.n_frames = kwargs.pop("n_frames", None)
        self.encoder_embed_dim = kwargs.pop("encoder_embed_dim", None)
        self.encoder_depth = kwargs.pop("encoder_depth", None)
        self.encoder_num_heads = kwargs.pop("encoder_num_heads", None)
        self.decoder_embed_dim = kwargs.pop("decoder_embed_dim", None)
        self.decoder_depth = kwargs.pop("decoder_depth", None)
        self.decoder_num_heads = kwargs.pop("decoder_num_heads", None)
        self.mlp_ratio = kwargs.pop("mlp_ratio", None)
        self.qkv_bias = kwargs.pop("qkv_bias", None)
        self.qk_scale = kwargs.pop("qk_scale", None)
        self.drop_rate = kwargs.pop("drop_rate", None)
        self.attn_drop_rate = kwargs.pop("attn_drop_rate", None)
        self.norm_layer = kwargs.pop("norm_layer", None)
        self.init_values = kwargs.pop("init_values", None)
        self.tubelet_size = kwargs.pop("tubelet_size", None)
        self.as_feature_extractor = kwargs.pop("as_feature_extractor", True)

        super().__init__(**kwargs)
