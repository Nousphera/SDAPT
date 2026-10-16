import math

import pytest
import torch

from sdapt.objectives import dino_loss, ibot_loss, koleo_loss, moco_v3_loss, sinkhorn_knopp


def test_sinkhorn_is_stable_and_balanced():
    logits = torch.tensor([[10000., 9999.], [9999., 10000.]])
    result = sinkhorn_knopp(logits, temperature=.04)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.sum(1), torch.ones(2))
    torch.testing.assert_close(result.sum(0), torch.ones(2))
    assert result[0, 0] > .999


def test_dino_excludes_same_global_view_and_includes_locals():
    targets = [torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]])]
    students = [torch.tensor([[0., 10.]]), torch.tensor([[10., 0.]])]
    assert dino_loss(students, targets, student_temperature=1).item() < .001
    value = dino_loss(students + [torch.zeros(1, 2)], targets, student_temperature=1)
    assert value.item() == pytest.approx(math.log(2) / 2, abs=1e-4)


def test_ibot_weights_each_image_equally_and_ignores_unmasked():
    mask = torch.tensor([[True, False, False], [True, True, True]])
    logits = torch.tensor([[0., 0.], [0., 10.], [0., 10.], [0., 10.]], requires_grad=True)
    targets = torch.tensor([[1., 0.], [0., 1.], [0., 1.], [0., 1.]])
    loss = ibot_loss(logits, targets, mask, student_temperature=1)
    assert loss.item() == pytest.approx(math.log(2) / 2, abs=1e-4)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_moco_cross_view_matching_and_detached_keys():
    q1 = torch.eye(3, requires_grad=True)
    q2 = torch.eye(3, requires_grad=True)
    k1 = torch.eye(3, requires_grad=True)
    k2 = torch.eye(3, requires_grad=True)
    correct = moco_v3_loss(q1, q2, k1, k2, temperature=.1)
    incorrect = moco_v3_loss(q1, q2, k1.roll(1, 0), k2.roll(1, 0), temperature=.1)
    assert correct < incorrect
    correct.backward()
    assert q1.grad is not None and k1.grad is None


def test_koleo_singleton_is_zero_and_duplicates_are_finite():
    assert koleo_loss(torch.ones(1, 3)).item() == 0
    assert torch.isfinite(koleo_loss(torch.ones(3, 3)))


def test_sinkhorn_rejects_nonfinite():
    with pytest.raises(FloatingPointError):
        sinkhorn_knopp(torch.tensor([[float('nan'), 0.]]), .07)
