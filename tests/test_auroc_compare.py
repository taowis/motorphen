import importlib.util
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location(
    "cmp", Path(__file__).resolve().parents[1] / "scripts" / "09_compare_auroc_e2.py")
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)


def test_delong_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    y = rng.random(60) < 0.4
    a = np.round(rng.normal(y * 0.5, 1), 1)
    b = rng.normal(y * 1.0, 1)
    auc, cov = cmp.delong(y, np.vstack([a, b]))
    assert np.allclose(auc, [roc_auc_score(y, a), roc_auc_score(y, b)])
    assert cov[0, 0] > 0 and cov[1, 1] > 0


def test_identical_scores_give_zero_difference():
    rng = np.random.default_rng(2)
    y = rng.random(40) < 0.5
    a = rng.normal(size=40)
    auc, cov = cmp.delong(y, np.vstack([a, a]))
    assert np.isclose(auc[0], auc[1])
    assert np.isclose(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 0)
