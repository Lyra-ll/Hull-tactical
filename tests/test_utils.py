import pytest

torch = pytest.importorskip("torch", reason="utils relies on PyTorch tensors")

import utils


def test_resolve_loss_weights_defaults_to_uniform():
    device = torch.device("cpu")
    weights = utils._resolve_loss_weights({}, 3, device)
    expected = torch.full((3,), 1 / 3, device=device)
    assert torch.allclose(weights, expected)


def test_resolve_loss_weights_uses_custom_values():
    device = torch.device("cpu")
    custom = [0.1, 0.2]
    weights = utils._resolve_loss_weights({'loss_weights': custom}, 2, device)
    assert torch.allclose(weights, torch.tensor(custom, dtype=torch.float32, device=device))


def test_resolve_loss_weights_handles_length_mismatch():
    device = torch.device("cpu")
    weights = utils._resolve_loss_weights({'loss_weights': [0.5, 0.3, 0.2]}, 1, device)
    expected = torch.tensor([1.0], device=device)
    assert torch.allclose(weights, expected)
