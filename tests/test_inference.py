import numpy as np
import torch

from motorphen.evaluate import bootstrap_equivalence, cycles, rmse
from motorphen.inference import badness, predict_last, reconstruct, select_joints
from motorphen.model import GaitMAE


def test_badness_zero_when_identical_and_grows_with_deviation():
    base = torch.zeros(5, 3)
    rom = torch.tensor([1.0, 0.5, 0.5])
    assert torch.allclose(badness(base, base, rom), torch.zeros(5), atol=1e-6)
    small = badness(base, base + torch.tensor([0.05, 0, 0]), rom)
    large = badness(base, base + torch.tensor([0.5, 0, 0]), rom)
    assert (large > small).all() and (small > 0).all()
    assert (large <= 1).all()


def test_badness_rotation_about_long_axis_is_geometrically_silent():
    # axial rotation does not move the long axis, so E_geom and B are zero
    base = torch.zeros(3, 3)
    assert torch.allclose(badness(base, base + torch.tensor([0, 0, 0.7]), torch.ones(3)),
                          torch.zeros(3), atol=1e-6)


def test_select_joints_floor_and_k():
    scores = {1: 0.5, 2: 0.9, 3: 0.1}
    assert select_joints(scores, None, 2) == [2, 1]
    assert select_joints(scores, {1: 0.6, 2: 0.2, 3: 0.0}, 2) == [2, 3]
    assert select_joints(scores, {1: 1, 2: 1, 3: 1}, 2) == []


def test_predict_last_and_reconstruct_shapes():
    torch.manual_seed(0)
    model = GaitMAE(enc_layers=1, dec_layers=1).eval()
    seq = torch.rand(20, 12, 3) - 0.5
    assert predict_last(model, seq, hide=[3]).shape == (14, 12, 3)
    assert torch.equal(reconstruct(model, seq, []), seq)
    out = reconstruct(model, seq, [8])
    assert out.shape == seq.shape
    assert torch.equal(out[:6], seq[:6]) and not torch.equal(out[6:], seq[6:])


def test_cycles_normalised_and_unwrapped():
    t = np.arange(91)
    series = 170 + 20 * np.sin(2 * np.pi * t / 30)        # crosses 180
    series = (series + 180) % 360 - 180                   # wrapped
    c = cycles(series, [0, 30, 60, 90])
    assert c.shape == (3, 100)
    assert np.ptp(c) < 45                                 # no 360-degree jumps


def test_rmse_wraps_angles():
    assert rmse([179.0], [-179.0]) == 2.0


def test_bootstrap_equivalence():
    rng = np.random.default_rng(0)
    assert bootstrap_equivalence(rng.normal(0, 0.3, 30))["equivalent"]
    assert not bootstrap_equivalence(rng.normal(3, 0.3, 30))["equivalent"]
