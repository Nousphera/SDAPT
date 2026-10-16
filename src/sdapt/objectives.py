"""DINO/iBOT, KoLeo and MoCo-v3 objectives, without CUDA-only extensions.

These are independent implementations of the mathematical objectives in DINOv2
(Oquab et al., 2024) and MoCo-v3 (Chen et al., 2021). Teacher assignments use
log-domain Sinkhorn to remain finite at low temperatures.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def _positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


@torch.no_grad()
def sinkhorn_knopp(logits: Tensor, temperature: float = .07, iterations: int = 3) -> Tensor:
    """Return balanced teacher probabilities for the complete server batch.

    Rows are samples and columns are prototypes. All clients must have been
    concatenated before calling this function; this is not a distributed gather.
    """
    _positive(temperature, 'temperature')
    if logits.ndim != 2 or min(logits.shape) == 0 or iterations < 1:
        raise ValueError('Sinkhorn needs nonempty [samples, prototypes] logits and iterations >= 1')
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Nonfinite teacher logits')
    log_q = logits.detach().float().T / temperature
    log_q -= torch.logsumexp(log_q.reshape(-1), 0)
    prototypes, samples = log_q.shape
    for _ in range(iterations):
        log_q -= torch.logsumexp(log_q, dim=1, keepdim=True) + math.log(prototypes)
        log_q -= torch.logsumexp(log_q, dim=0, keepdim=True) + math.log(samples)
    return (log_q + math.log(samples)).exp().T


def dino_loss(student_views: Sequence[Tensor], teacher_global_targets: Sequence[Tensor],
              student_temperature: float = .1) -> Tensor:
    """Mean cross-view CE; first two student views are the matching globals."""
    _positive(student_temperature, 'student_temperature')
    if len(student_views) < 2 or len(teacher_global_targets) != 2:
        raise ValueError('DINO needs two globals followed by optional local crops')
    terms = []
    for student_index, logits in enumerate(student_views):
        log_probs = F.log_softmax(logits.float() / student_temperature, dim=-1)
        for teacher_index, target in enumerate(teacher_global_targets):
            if student_index != teacher_index:
                if logits.shape != target.shape:
                    raise ValueError('All DINO views must have aligned samples and prototypes')
                terms.append(-(target.detach().float() * log_probs).sum(-1).mean())
    return torch.stack(terms).mean()


def ibot_loss(student_masked_logits: Tensor, teacher_masked_targets: Tensor, mask: Tensor,
              student_temperature: float = .1) -> Tensor:
    """Masked-patch CE, normalized per image then averaged across global views.

    Inputs contain only masked patches in row-major mask order. Images with no
    selected patches contribute zero, matching DINOv2's image normalization.
    """
    _positive(student_temperature, 'student_temperature')
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError('mask must be a boolean [global_images, patches] tensor')
    if student_masked_logits.shape != teacher_masked_targets.shape:
        raise ValueError('Student and teacher patch logits must align')
    if student_masked_logits.shape[0] != int(mask.sum()):
        raise ValueError('Patch logits do not match mask cardinality')
    if student_masked_logits.shape[0] == 0:
        return student_masked_logits.sum() * 0
    losses = -(teacher_masked_targets.detach().float() * F.log_softmax(
        student_masked_logits.float() / student_temperature, dim=-1)).sum(-1)
    weights = mask.sum(-1).clamp_min(1).float().reciprocal()[:, None].expand_as(mask)[mask]
    return (losses * weights).sum() / mask.shape[0]


def koleo_loss(features: Tensor, eps: float = 1e-8) -> Tensor:
    """Negative log nearest-neighbor distance on L2-normalized CLS features."""
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError('KoLeo requires a nonempty [batch, features] tensor')
    if len(features) < 2:
        return features.sum() * 0
    x = F.normalize(features.float(), dim=-1, eps=eps)
    with torch.no_grad():
        similarities = x @ x.T
        similarities.fill_diagonal_(-float('inf'))
        neighbor = similarities.argmax(dim=1)
    distances = (x - x[neighbor]).square().sum(-1).clamp_min(eps * eps).sqrt()
    return -distances.log().mean()


def moco_v3_loss(q1: Tensor, q2: Tensor, k1: Tensor, k2: Tensor,
                 temperature: float = .2) -> Tensor:
    """Symmetric MoCo-v3 in-batch InfoNCE, including the official 2*T scale.

    There is no MoCo-v2 queue. Both view batches must use the same sample order.
    """
    _positive(temperature, 'temperature')
    if len({tuple(t.shape) for t in (q1, q2, k1, k2)}) != 1 or q1.ndim != 2 or len(q1) < 2:
        raise ValueError('MoCo-v3 requires aligned [batch>=2, features] views')
    labels = torch.arange(len(q1), device=q1.device)
    def direction(query: Tensor, key: Tensor) -> Tensor:
        logits = F.normalize(query.float(), dim=-1) @ F.normalize(key.detach().float(), dim=-1).T
        return F.cross_entropy(logits / temperature, labels) * (2 * temperature)
    return direction(q1, k2) + direction(q2, k1)
