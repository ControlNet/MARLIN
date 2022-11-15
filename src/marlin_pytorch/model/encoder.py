from torch import nn, Tensor
from torch.nn import ModuleList, LayerNorm

from .util import PatchEmbedding3d, Block
from .positional_embedding import SinCosPositionalEmbedding


class MarlinEncoder(nn.Module):

    def __init__(self, img_size=224, patch_size=16, n_frames=16, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 norm_layer="LayerNorm", init_values=0., tubelet_size=2):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_embedding = PatchEmbedding3d(
            input_size=(3, n_frames, img_size, img_size),
            patch_size=(tubelet_size, patch_size, patch_size),
            embedding=embed_dim
        )
        num_patches = (img_size // patch_size) * (img_size // patch_size) * (n_frames // tubelet_size)

        # sine-cosine positional embeddings
        self.pos_embedding = SinCosPositionalEmbedding((num_patches, embed_dim), dropout_rate=0.)

        if norm_layer == "LayerNorm":
            self.norm_layer = LayerNorm
            self.norm = self.norm_layer(embed_dim)
        else:
            raise NotImplementedError("Only LayerNorm is supported")

        self.blocks = ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=self.norm_layer,
                init_values=init_values)
            for _ in range(depth)
        ])

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # mask: (B, T, N) with boolean values, 0 -> masked, 1 -> visible
        assert len(x.shape) == 5, "x must be 5D"
        emb = self.patch_embedding(x)
        emb = self.pos_embedding(emb)
        b, _, c = emb.shape
        emb = emb[mask].view(b, -1, c)  # only visible patches are used
        emb = self.forward_features(emb)
        return emb

    def extract_features(self, x: Tensor, seq_mean_pool: bool) -> Tensor:
        x = self.patch_embedding(x)
        x = self.pos_embedding(x)
        for block in self.blocks:
            x = block(x)

        if seq_mean_pool:
            x = x.mean(dim=1)
        x = self.norm(x)
        return x
