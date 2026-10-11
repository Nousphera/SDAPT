import torch
import torch.nn as nn


class MaskedBCEWithLogitsLoss(nn.Module):
    """BCE-with-logits that ignores entries labeled -1 (uncertain)."""

    def __init__(self, ignore_index: int = -1):
        super().__init__()
        self.ignore_index = ignore_index
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, labels):
        mask = (labels != self.ignore_index).float()        
        loss = self.bce(logits, labels.clamp(min=0))     
        loss = loss * mask
        return loss.sum() / mask.sum().clamp(min=1)         