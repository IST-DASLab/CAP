import torch
import torch.nn as nn

from timm.models.vision_transformer import Attention


__all__ = [
    "split_qkv",
    "SplitAttention"
]


class SplitAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn[mask[:, None, None, :].repeat(1, self.num_heads, N, 1)] = torch.finfo(attn.dtype).min
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


@torch.no_grad()
def split_qkv(model: nn.Module):
    for module_name, module in model.named_modules():
        if isinstance(module, Attention):
            dim = module.qkv.in_features
            attention = SplitAttention(
                dim,
                num_heads=module.num_heads,
                qkv_bias=hasattr(module.qkv, 'bias') and module.qkv.bias is not None,
                attn_drop=module.attn_drop.p,
                proj_drop=module.proj_drop.p
            )

            for i, layer in enumerate([attention.q, attention.k, attention.v]):
                layer.weight.data =  module.qkv.weight.data[(i * dim):((i + 1) * dim), :]
                layer.bias.data = module.qkv.bias.data[(i * dim):((i + 1) * dim)]

            attention.proj = module.proj
            attention = attention.to(module.qkv.weight.device)
            # get parent module
            parent_name, name = module_name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_name)
            setattr(parent_module, name, attention)
    return model
