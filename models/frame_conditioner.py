from flash_attn import flash_attn_func
import torch.nn as nn
import torch


# Cross attention layer for computing attention across x and y
class CrossAttention(nn.Module):
    def __init__(self, dim=512, cross_dim=512, num_heads=8, dropout_p=0.0):
        super().__init__()

        # Layers for projecting Q, K, V
        self.q_proj = nn.Linear(dim, cross_dim * num_heads)
        self.k_proj = nn.Linear(dim, cross_dim * num_heads)
        self.v_proj = nn.Linear(dim, cross_dim * num_heads)

        self.num_heads = num_heads
        self.cross_dim = cross_dim
        self.dropout_p = dropout_p

    def forward(self, q, k, v):
        # Input: (batch_size, seqlen, dim)
        # Output: (batch_size, seqlen, num_heads, headdim)

        q_proj = self.q_proj(q).view(
            q.size(0), q.size(1), self.num_heads, self.cross_dim
        )
        k_proj = self.k_proj(k).view(
            k.size(0), k.size(1), self.num_heads, self.cross_dim
        )
        v_proj = self.v_proj(v).view(
            v.size(0), v.size(1), self.num_heads, self.cross_dim
        )

        # Using flash_attn_func for the attention mechanism
        out = flash_attn_func(
            q_proj,
            k_proj,
            v_proj,
            dropout_p=self.dropout_p,
            softmax_scale=1 / (self.cross_dim**0.5),
        )

        return out  # Output shape: (batch_size, seqlen, nheads, headdim)


# CrossAttentionBlock class
#
# Mimics a transformer block from CLIP
# This uses layer norms, residuals and MLP
class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim_x=512,
        dim_y=256,
        cross_dim=256,
        num_heads=8,
        mlp_hidden_dim=512 * 4,
        dropout_p=0.0,
    ):
        super().__init__()

        # Cross Attention Layer for x attending to y
        self.cross_attention = CrossAttention(
            dim=dim_x, cross_dim=cross_dim, num_heads=num_heads, dropout_p=dropout_p
        )

        # Layer Norms for x and y
        self.norm_1 = nn.LayerNorm(dim_x)
        self.norm_2 = nn.LayerNorm(dim_x)

        # MLP for x
        self.mlp_x = nn.Sequential(
            nn.Linear(dim_x, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, dim_x),
        )

    def forward(self, x, y):
        # Inputs:
        # x shape: (batch_size, seqlen_x, dim_x)
        # y shape: (batch_size, seqlen_y, dim_y)

        # Layer Norm -> Cross Attention
        attn_out_x = self.cross_attention(
            self.norm_1(x), self.norm_1(y), self.norm_1(y)
        )

        # Adding residual connection
        out_x = x + attn_out_x

        # Layer Norm -> MLP for x
        out_x = out_x + self.mlp_x(self.norm_2(out_x))

        return out_x


CLIP_EMBED_DIM = 512


# Adds two CrossAttentionBlocks to optionally conditon the video on initial and end frames
class FrameConditionEmbedding(nn.Module):
    def __init__(
        self,
        dim=512,
        cross_dim=256,
        num_heads=8,
        droput_cross_attention=0.0,
        mlp_hidden_dim=1024,
    ):
        super().__init__()

        self.initial_cross_attention = CrossAttentionBlock(
            dim=dim,
            cross_dim=cross_dim,
            num_heads=num_heads,
            dropout_p=droput_cross_attention,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self.final_cross_attention = CrossAttentionBlock(
            dim=dim,
            cross_dim=cross_dim,
            num_heads=num_heads,
            dropout_p=droput_cross_attention,
            mlp_hidden_dim=mlp_hidden_dim,
        )

    def forward(self, base_clip_embed, initial_clip_embed=None, final_clip_embed=None):
        # default args
        if base_clip_embed is None:
            base_clip_embed = torch.zeros(clip_embed_dim=CLIP_EMBED_DIM)

        if initial_clip_embed is None:
            initial_clip_embed = torch.zeros(clip_embed_dim=CLIP_EMBED_DIM)

        if final_clip_embed is not None:
            final_clip_embed = torch.zeros(clip_embed_dim=CLIP_EMBED_DIM)

        # compute layer graph
        x = initial_clip_embed
        x = self.initial_cross_attention(x, initial_clip_embed)
        x = self.final_cross_attention(x, final_clip_embed)

        return x
