import numpy as np
import pytest

from motorphen.preprocess import contiguous_segments, features, fill_gaps, resample, walking_span, windows


def test_fill_gaps_short_filled_long_kept():
    x = np.arange(20, dtype=float)
    x[3:5] = np.nan          # gap of 2
    x[10:16] = np.nan        # gap of 6
    y = fill_gaps(x[:, None], max_gap=3)[:, 0]
    assert np.allclose(y[3:5], [3, 4])
    assert np.isnan(y[10:16]).all()


def test_fill_gaps_leading_nan_not_extrapolated():
    x = np.array([np.nan, np.nan, 2.0, 3.0])
    y = fill_gaps(x[:, None], max_gap=5)[:, 0]
    assert np.isnan(y[:2]).all()


def test_segments_split_at_nan_and_respect_min_len():
    x = np.ones((30, 2))
    x[10] = np.nan
    assert contiguous_segments(x, 5) == [(0, 10), (11, 30)]
    assert contiguous_segments(x, 15) == [(11, 30)]


def test_walking_span_uses_both_sides_and_clips():
    ev = {"Left Foot Strike": [10, 120, 400], "Right Foot Strike": [60, 170]}
    assert walking_span(ev, 300) == (10, 171)
    assert walking_span({"Left Foot Strike": [5]}, 100) is None


def test_resample_length_and_shape():
    t = np.linspace(0, 2, 200, endpoint=False)
    x = np.stack([np.sin(2 * np.pi * t)] * 3, axis=1)
    y = resample(x, 100, 30)
    assert y.shape == (60, 3)
    # a 1 Hz sine survives the resampling (ignore edges)
    t30 = np.arange(60) / 30
    assert np.max(np.abs(y[5:-5, 0] - np.sin(2 * np.pi * t30[5:-5]))) < 0.02


def test_features_shape_and_rotation_columns_orthonormal():
    rng = np.random.default_rng(0)
    a = rng.uniform(-60, 60, size=(4, 12, 3))
    f = features(a)
    assert f.shape == (4, 12, 12)
    assert np.allclose(np.arctan2(f[..., :3], f[..., 3:6]), np.deg2rad(a))
    c1, c2 = f[..., 6:9], f[..., 9:12]
    assert np.allclose((c1 * c1).sum(-1), 1) and np.allclose((c2 * c2).sum(-1), 1)
    assert np.allclose((c1 * c2).sum(-1), 0, atol=1e-12)


def test_features_zero_angles_identity():
    f = features(np.zeros((1, 3)))
    assert np.allclose(f[0, 6:], [1, 0, 0, 0, 1, 0])


def test_windows():
    x = np.arange(10)[:, None]
    w = windows(x, 7)
    assert w.shape == (4, 7, 1)
    assert w[-1, -1, 0] == 9
    assert windows(x, 11).shape[0] == 0
    assert windows(x, 7, step=3).shape[0] == 2


def test_labels_strip_subject_prefix():
    from motorphen.c3d_io import _labels
    params = {"LABELS": {"value": ["NAME:LHipAngles", "RKneeAngles"]},
              "LABELS2": {"value": ["X:LANK"]}}
    assert _labels(params) == ["LHipAngles", "RKneeAngles", "LANK"]
