# --------------------------------------------------------
# References:
# DinoV2: https://github.com/facebookresearch/dinov2
# Dino: https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
# TIMM: https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py
# --------------------------------------------------------

from typing import Callable, Optional

from torch import Tensor, nn
from .lora_layers_util import LoRALinearLayer


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def init_lora(self, lora_rank, **kwargs):
        in_features, hidden_features = self.fc1.in_features, self.fc1.out_features
        out_features = self.fc2.out_features
        self.fc1 = LoRALinearLayer(in_features, hidden_features, r=lora_rank // 4)
        self.fc2 = LoRALinearLayer(hidden_features, out_features, r=lora_rank // 4)  # Since 4 is mlp ratio

    def deinit_lora(self, **kwargs):
        # Nothing to do here
        pass

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
