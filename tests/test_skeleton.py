import numpy as np
from scipy.spatial.transform import Rotation

from motorphen import skeleton
from motorphen.c3d_io import JOINTS


def random_frames(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return {s: (Rotation.random(n, random_state=k).as_matrix(), rng.normal(size=(n, 3)) * 100)
            for k, s in enumerate(skeleton.SEGMENTS)}


def test_relative_unchanged_when_angles_equal():
    rel = Rotation.random(4, random_state=1).as_matrix()
    ang = np.deg2rad(np.random.default_rng(2).uniform(-40, 40, (4, 3)))
    for joint in skeleton.CONVENTION:
        assert np.allclose(skeleton.relative(joint, rel, ang, ang), rel)


def test_swap_with_original_angles_keeps_pose():
    frames = random_frames()
    centres = skeleton.fit(frames)
    ang = np.deg2rad(np.random.default_rng(3).uniform(-30, 30, (len(JOINTS), 3)))
    p0 = skeleton.points(*skeleton.pose(frames, centres, 5), centres)
    p1 = skeleton.points(*skeleton.pose(frames, centres, 5, ang, ang.copy()), centres)
    assert all(np.allclose(p0[k], p1[k]) for k in p0)


def test_knee_change_moves_only_that_shank():
    frames = random_frames()
    centres = skeleton.fit(frames)
    ang = np.zeros((len(JOINTS), 3))
    rec = ang.copy()
    rec[JOINTS.index("r_knee"), 0] = np.deg2rad(20)
    p0 = skeleton.points(*skeleton.pose(frames, centres, 7, ang, ang.copy()), centres)
    p1 = skeleton.points(*skeleton.pose(frames, centres, 7, ang, rec), centres)
    moved = {k for k in p0 if not np.allclose(p0[k], p1[k])}
    assert moved == {"r_ank", "r_toe"}
