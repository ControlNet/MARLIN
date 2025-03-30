from marlin_huggingface import MarlinConfig


vit_base_config = MarlinConfig(
    img_size=224,
    patch_size=16,
    n_frames=16,
    mlp_ratio=4.0,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    norm_layer="LayerNorm",
    init_values=0.0,
    tubelet_size=2,
    encoder_embed_dim=768,
    encoder_depth=12,
    encoder_num_heads=12,
    decoder_embed_dim=384,
    decoder_depth=4,
    decoder_num_heads=6,
)
