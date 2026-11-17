"""Tests for src/trajectory.py. Run from the repo root:  python tests/test_trajectory.py"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import trajectory as traj


def test_quintic_boundary_conditions():
    q0 = np.array([0.1, -0.5, 1.2, 0.0, -1.0, 0.5])
    q1 = np.array([0.8, 0.2, -0.4, 1.5, 0.3, -0.7])
    dur = 3.0
    times, q, qd, qdd = traj.generate_trajectory(q0, q1, duration=dur, dt=0.01, method="quintic")
    
    # Start conditions
    assert np.allclose(q[0], q0, atol=1e-7), "Start position mismatch"
    assert np.allclose(qd[0], np.zeros(6), atol=1e-7), "Start velocity non-zero"
    assert np.allclose(qdd[0], np.zeros(6), atol=1e-7), "Start acceleration non-zero"
    
    # End conditions
    assert np.allclose(q[-1], q1, atol=1e-7), "End position mismatch"
    assert np.allclose(qd[-1], np.zeros(6), atol=1e-7), "End velocity non-zero"
    assert np.allclose(qdd[-1], np.zeros(6), atol=1e-7), "End acceleration non-zero"


def test_cubic_boundary_conditions():
    q0 = np.zeros(6)
    q1 = np.ones(6)
    dur = 2.0
    times, q, qd, _ = traj.generate_trajectory(q0, q1, duration=dur, dt=0.01, method="cubic")
    assert np.allclose(q[0], q0, atol=1e-7)
    assert np.allclose(qd[0], np.zeros(6), atol=1e-7)
    assert np.allclose(q[-1], q1, atol=1e-7)
    assert np.allclose(qd[-1], np.zeros(6), atol=1e-7)


def test_multisegment_continuity():
    wps = [np.zeros(6), np.ones(6) * 0.5, -np.ones(6) * 0.3]
    durs = [1.5, 2.0]
    times, q, qd, qdd = traj.plan_multisegment_trajectory(wps, durs, dt=0.02)
    assert len(times) == len(q)
    assert np.allclose(q[0], wps[0])
    assert np.allclose(q[-1], wps[-1])
    assert times[-1] >= 3.49 and times[-1] <= 3.51


if __name__ == "__main__":
    test_quintic_boundary_conditions()
    print("PASS test_quintic_boundary_conditions")
    test_cubic_boundary_conditions()
    print("PASS test_cubic_boundary_conditions")
    test_multisegment_continuity()
    print("PASS test_multisegment_continuity")
    print("All trajectory tests passed!")
