"""Client-side ("head") module for MoCo split learning.

Mirrors ``models/dinov2/.../DinoHead``: this module owns the patch embedding,
cls token and positional embedding, and returns *prepared tokens* that the
server-side backbone (``MoCoBackbone``) consumes.

Differences vs. DinoHead:
  * no ``mask_token`` / iBOT masking path (MoCo has no masked-patch objective)
  * ``freeze_patch_embed()`` implements the MoCo v3 stability trick: the patch
    projection is kept at its random (or pretrained) init and never receives
    gradients. This is what stops the mid-training loss spikes that plague
    contrastive ViT pretraining.
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from models.dinov2.layers import PatchEmbed

logger = logging.getLogger("moco")


class MoCoHead(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=14,
        in_chans=3,
        embed_dim=768,
        embed_layer=PatchEmbed,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        freeze_patch_embed=True,
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            logger.debug("Ignoring unused MoCoHead kwargs: %s", sorted(kwargs))

        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        self.init_weights()

        self._patch_embed_frozen = False
        if freeze_patch_embed:
            self.freeze_patch_embed()

    # ------------------------------------------------------------------ init

    def init_weights(self):
        trunc_normal_(self.patch_embed.proj.weight, std=0.02)
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)

        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)

    def freeze_patch_embed(self):
        """MoCo v3 trick: stop gradients at the patch projection."""
        for p in self.patch_embed.parameters():
            p.requires_grad = False
        self._patch_embed_frozen = True
        logger.info("MoCoHead: patch_embed frozen (MoCo v3 stability trick)")

    def unfreeze_patch_embed(self):
        for p in self.patch_embed.parameters():
            p.requires_grad = True
        self._patch_embed_frozen = False

    def requires_any_grad(self) -> bool:
        return any(p.requires_grad for p in self.parameters())

    # -------------------------------------------------------------- forward

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens(self, x):
        _, _, w, h = x.shape
        x = self.patch_embed(x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        return x

    def forward_features(self, x):
        if isinstance(x, list):
            return [self.prepare_tokens(xi) for xi in x]
        return self.prepare_tokens(x)

    def forward(self, x):
        return self.forward_features(x)