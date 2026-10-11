import math
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


class Evaluator:
    def reset(self): raise NotImplementedError
    def update(self, logits, labels, client_id=None): raise NotImplementedError
    def compute(self): raise NotImplementedError  # -> dict | None