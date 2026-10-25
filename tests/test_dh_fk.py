"""Tests for src/dh_fk.py. Run from the repo root:  python tests/test_dh_fk.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import dh_fk as fk

rng = np.random.default_rng(0)
LIMIT = 2 * np.pi


def random_q():
    return rng.uniform(-LIMIT, LIMIT, size=6)


def rot_log_vec(R):
    """Axis-angle vector of a small rotation matrix (for the numerical Jacobian)."""
    return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def test_zero_pose_matches_known_ur5_home():
    p, _ = fk.position_and_rotation(np.zeros(6))
    expected = np.array([-0.81725, -0.19145, -0.005491])
    assert np.allclose(p, expected, atol=1e-4), f"got {p}, expected {expected}"


def test_rotation_is_valid():
    for _ in range(200):
        R = fk.forward_kinematics(random_q())[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_last_row_is_homogeneous():
    T = fk.forward_kinematics(random_q())
    assert np.allclose(T[3], [0, 0, 0, 1])


def test_reach_never_exceeds_arm_length():
    # Distance from shoulder (joint 2 origin) to flange is bounded by the arm's reach.
    reach = abs(fk.DH_TABLE[1, 0]) + abs(fk.DH_TABLE[2, 0]) + fk.DH_TABLE[3, 2] \
        + fk.DH_TABLE[4, 2] + fk.DH_TABLE[5, 2]
    for _ in range(200):
        frames = fk.all_frames(random_q())
        dist = np.linalg.norm(frames[-1][:3, 3] - frames[1][:3, 3])
        assert dist <= reach + 1e-9


def test_all_frames_has_seven_entries_and_last_matches_fk():
    q = random_q()
    frames = fk.all_frames(q)
    assert len(frames) == 7
    assert np.allclose(frames[-1], fk.forward_kinematics(q))


def test_joint1_rotation_spins_about_base_z():
    q = np.array([0.3, -0.7, 1.1, 0.2, -0.4, 0.9])
    q2 = q.copy()
    q2[0] += 0.5
    p1, _ = fk.position_and_rotation(q)
    p2, _ = fk.position_and_rotation(q2)
    # Rotating joint 1 keeps height and distance from the base z axis unchanged.
    assert np.isclose(p1[2], p2[2], atol=1e-9)
    assert np.isclose(np.hypot(p1[0], p1[1]), np.hypot(p2[0], p2[1]), atol=1e-9)


def test_jacobian_matches_numerical():
    h = 1e-6
    for _ in range(50):
        q = random_q()
        T0 = fk.forward_kinematics(q)
        J_num = np.zeros((6, 6))
        for i in range(6):
            dq = np.zeros(6)
            dq[i] = h
            T1 = fk.forward_kinematics(q + dq)
            J_num[:3, i] = (T1[:3, 3] - T0[:3, 3]) / h
            dR = T1[:3, :3] @ T0[:3, :3].T
            J_num[3:, i] = rot_log_vec(dR) / h
        assert np.allclose(fk.jacobian(q), J_num, atol=1e-4)


def test_singularity_has_low_manipulability():
    # Wrist singularity: joint 5 = 0 lines up joints 4 and 6.
    q_sing = np.array([0.4, -1.2, 1.5, 0.3, 0.0, 0.8])
    q_ok = np.array([0.4, -1.2, 1.5, 0.3, 1.2, 0.8])
    assert fk.manipulability(q_sing) < 1e-6
    assert fk.manipulability(q_ok) > 1e-3


def test_wrong_input_size_raises():
    try:
        fk.forward_kinematics([0, 0, 0])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"All {len(tests)} tests passed")
