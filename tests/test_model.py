import numpy as np
import torch

from motorphen.masking import joint_mask, sample_masks, structured_prob
from motorphen.model import GaitMAE, angles_from_pred, token_features
from motorphen.preprocess import features


def test_token_features_match_numpy_version():
    rng = np.random.default_rng(1)
    a = rng.uniform(-60, 60, size=(5, 12, 3))
    t = token_features(torch.tensor(np.deg2rad(a))).numpy()
    assert np.allclose(t, features(a), atol=1e-10)


def test_forward_shape_and_angle_recovery():
    torch.manual_seed(0)
    model = GaitMAE(enc_layers=1, dec_layers=1)
    a = torch.rand(3, 12, 7, 3) - 0.5
    out = model(a, torch.zeros(3, 12, 7, dtype=torch.bool))
    assert out.shape == (3, 12, 7, 12)
    assert angles_from_pred(out).shape == (3, 12, 7, 3)


def test_masked_tokens_do_not_leak():
    torch.manual_seed(0)
    model = GaitMAE(enc_layers=2, dec_layers=1).eval()
    a = torch.rand(2, 12, 7, 3) - 0.5
    m = joint_mask(2, 12, 7, [8])          # hide l_knee at all frames
    m[:, 3, 2:5] = True                    # and part of l_elbow
    b = a.clone()
    b[m] = torch.rand(int(m.sum()), 3)     # change only hidden values
    with torch.no_grad():
        assert torch.allclose(model(a, m), model(b, m), atol=1e-5)
        # sanity: changing a visible value does change the output
        c = a.clone()
        c[:, 0, 6] += 0.5
        assert not torch.allclose(model(a, m), model(c, m), atol=1e-5)


def test_masks_shape_and_bounds():
    g = torch.Generator().manual_seed(0)
    for epoch in (0, 30, 100):
        m = sample_masks(512, 12, 7, epoch, gen=g)
        assert m.shape == (512, 12, 7) and m.dtype == torch.bool
        flat = m.flatten(1)
        assert flat.any(1).all() and (~flat).any(1).all()


def test_random_masks_early_epochs_about_half():
    g = torch.Generator().manual_seed(0)
    m = sample_masks(4000, 12, 7, 0, gen=g)
    assert abs(m.float().mean().item() - 0.5) < 0.02


def test_structured_masks_later_epochs_sparser():
    g = torch.Generator().manual_seed(0)
    m = sample_masks(4000, 12, 7, 200, gen=g)
    assert m.float().mean().item() < 0.3


def test_curriculum_schedule():
    assert structured_prob(0) == 0 and structured_prob(4) == 0
    assert 0 < structured_prob(30) < 0.9
    assert structured_prob(60) == 0.9 == structured_prob(249)


def test_full_window_spans_occur_and_are_coherent():
    g = torch.Generator().manual_seed(0)
    m = sample_masks(4000, 12, 7, 200, gen=g)
    same = (m == m[:, :, :1]).all(dim=(1, 2))
    assert 0.2 < same.float().mean().item() < 0.6   # 25% full spans plus chance agreement
