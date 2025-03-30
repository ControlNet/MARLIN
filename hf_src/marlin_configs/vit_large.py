from marlin_huggingface import MarlinConfig


vit_large_config = MarlinConfig(
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
    encoder_embed_dim=1024,
    encoder_depth=24,
    encoder_num_heads=16,
    decoder_embed_dim=512,
    decoder_depth=12,
    decoder_num_heads=8,
)
