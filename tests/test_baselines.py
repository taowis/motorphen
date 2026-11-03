import numpy as np

from motorphen.baselines import PCABaseline, normative_reference, peak_knee_flexion, rmse_side, zscore_side
from motorphen.c3d_io import JOINTS


def gait(n=150, knee_amp=30.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 30
    a = rng.normal(0, 0.5, size=(n, len(JOINTS), 3))
    for side in ("l", "r"):
        a[:, JOINTS.index(f"{side}_hip"), 0] += 20 * np.sin(2 * np.pi * t)
        a[:, JOINTS.index(f"{side}_knee"), 0] += 30 + knee_amp * np.sin(2 * np.pi * t)
    return a


STRIKES = list(range(0, 150, 30))


def test_reference_scores_low_for_healthy_and_high_for_stiff_knee():
    healthy = [(gait(seed=s), STRIKES) for s in range(5)]
    ref = normative_reference({"left": healthy, "right": healthy})
    normal = [(gait(seed=10), STRIKES)]
    stiff = gait(seed=11)
    stiff[:, JOINTS.index("r_knee"), 0] = 30 + 5 * np.sin(2 * np.pi * np.arange(150) / 30)
    stiff = [(stiff, STRIKES)]
    assert zscore_side(stiff, "right", ref) > 3 * zscore_side(normal, "right", ref)
    assert rmse_side(stiff, "right", ref) > rmse_side(normal, "right", ref)
    assert peak_knee_flexion(stiff, "right") < peak_knee_flexion(stiff, "left")


def test_pca_error_highest_on_perturbed_joint():
    train = np.concatenate([gait(seed=s) for s in range(5)])
    p = PCABaseline(var=0.9).fit(train)
    x = gait(seed=20)
    x[:, JOINTS.index("l_knee"), 0] += np.random.default_rng(1).normal(0, 25, 150)
    err = p.joint_error(x)
    assert int(np.argmax(err)) == JOINTS.index("l_knee")
    assert p.side_score([x], "left") > p.side_score([x], "right")
