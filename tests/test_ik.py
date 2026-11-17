"""Tests for src/ik_analytical.py and src/ik_dls.py. Run from the repo root:  python tests/test_ik.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import dh_fk as fk
import ik_analytical as ana
import ik_dls as dls

rng = np.random.default_rng(42)
LIMIT = np.pi


def random_q():
    return rng.uniform(-LIMIT, LIMIT, size=6)


def wrap(a):
    return (np.asarray(a, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def pose_error(T_a, T_b):
    """(position error [m], rotation error [rad]) between two poses.

    The angle uses atan2(sin, cos) so it stays accurate near zero, where
    arccos((trace - 1) / 2) loses precision to rounding.
    """
    dp = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    dR = T_a[:3, :3] @ T_b[:3, :3].T
    cos_t = np.clip((np.trace(dR) - 1) / 2, -1, 1)
    v = np.array([dR[2, 1] - dR[1, 2],
                  dR[0, 2] - dR[2, 0],
                  dR[1, 0] - dR[0, 1]]) / 2
    ang = float(np.arctan2(np.linalg.norm(v), cos_t))
    return dp, ang


def report(name, failures):
    for line in failures:
        print(f"  FAILURE {line}")
    if failures:
        raise AssertionError(f"{name}: {len(failures)} failure(s), logged above")


def test_analytical_roundtrip_recovers_pose_and_joints():
    name = "test_analytical_roundtrip_recovers_pose_and_joints"
    failures = []
    for _ in range(200):
        q = random_q()
        T = fk.forward_kinematics(q)
        try:
            q_ik = ana.inverse_kinematics(T, q_prefer=q)
            dp, ang = pose_error(fk.forward_kinematics(q_ik), T)
            worst = float(np.max(np.abs(wrap(q_ik - q))))
            if dp > 1e-9 or ang > 1e-9:
                failures.append(f"pose error dp={dp:.2e} ang={ang:.2e} for q={np.round(q, 4)}")
            if worst > 1e-6:
                failures.append(f"joint recovery {worst:.2e} for q={np.round(q, 4)}")
        except ValueError as exc:
            failures.append(f"ValueError({exc}) for q={np.round(q, 4)}")
    report(name, failures)


def test_analytical_all_solutions_match_target():
    name = "test_analytical_all_solutions_match_target"
    failures = []
    for _ in range(20):
        T = fk.forward_kinematics(random_q())
        try:
            sols = ana.all_solutions(T)
        except ValueError as exc:
            failures.append(f"ValueError({exc}) for reachable pose")
            continue
        if not 1 <= len(sols) <= 8:
            failures.append(f"got {len(sols)} solutions (expected 1..8)")
        for s in sols:
            dp, ang = pose_error(fk.forward_kinematics(s), T)
            if dp > 1e-9 or ang > 1e-9:
                failures.append(f"solution {np.round(s, 4)} pose error "
                                f"dp={dp:.2e} ang={ang:.2e}")
    report(name, failures)


def test_analytical_best_solution_follows_preference():
    name = "test_analytical_best_solution_follows_preference"
    failures = []
    for _ in range(20):
        q = random_q()
        T = fk.forward_kinematics(q)
        sols = ana.all_solutions(T)
        for prefer in (random_q(), None):
            q_sel = ana.inverse_kinematics(T, q_prefer=prefer)
            ref = np.zeros(6) if prefer is None else prefer
            costs = [float(np.sum(wrap(s - ref) ** 2)) for s in sols]
            best = sols[int(np.argmin(costs))]
            if not np.allclose(q_sel, best, atol=1e-12):
                failures.append(f"selected {np.round(q_sel, 4)} but closest "
                                f"to {np.round(ref, 4)} is {np.round(best, 4)}")
            dp, _ = pose_error(fk.forward_kinematics(q_sel), T)
            if dp > 1e-9:
                failures.append(f"selected solution off pose by {dp:.2e} m")
    report(name, failures)


def test_analytical_handles_wrist_singularity():
    name = "test_analytical_handles_wrist_singularity"
    failures = []
    q_sing = np.array([0.4, -1.2, 1.5, 0.3, 0.0, 0.8])
    T = fk.forward_kinematics(q_sing)
    try:
        q_ik = ana.inverse_kinematics(T, q_prefer=q_sing)
        dp, ang = pose_error(fk.forward_kinematics(q_ik), T)
        if dp > 1e-9 or ang > 1e-9:
            failures.append(f"wrist-singular pose error dp={dp:.2e} ang={ang:.2e}")
    except ValueError as exc:
        failures.append(f"ValueError({exc}) for wrist-singular pose")
    report(name, failures)


def test_analytical_rejects_unreachable_targets():
    R = fk.forward_kinematics(random_q())[:3, :3]
    cases = {
        "outside shoulder cone": [0.05, 0.0, 0.10],
        "beyond two-link reach": [1.0, 0.0, 0.089159],
    }
    for name_case, pos in cases.items():
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        try:
            ana.inverse_kinematics(T)
        except ValueError as exc:
            print(f"  rejected unreachable ({name_case}): {exc}")
            continue
        raise AssertionError(f"{name_case}: expected ValueError for {pos}")


def test_analytical_rejects_bad_input_shape():
    try:
        ana.inverse_kinematics(np.zeros((3, 3)))
    except ValueError:
        return
    raise AssertionError("expected ValueError for a 3x3 input")


def test_dls_roundtrip_from_zero_start():
    name = "test_dls_roundtrip_from_zero_start"
    failures = []
    for _ in range(30):
        q = random_q()
        T = fk.forward_kinematics(q)
        try:
            q_ik, info = dls.inverse_kinematics(T, return_info=True)
            if not info["converged"]:
                failures.append(f"no convergence: {info} for q={np.round(q, 4)}")
            if info["position_error"] > 1e-7 or info["rotation_error"] > 1e-7:
                failures.append(f"error pos={info['position_error']:.2e} "
                                f"rot={info['rotation_error']:.2e} for q={np.round(q, 4)}")
        except RuntimeError as exc:
            failures.append(f"RuntimeError({exc}) for q={np.round(q, 4)}")
    report(name, failures)


def test_dls_converges_from_perturbed_start():
    name = "test_dls_converges_from_perturbed_start"
    failures = []
    for _ in range(30):
        q = random_q()
        q0 = wrap(q + rng.uniform(-0.4, 0.4, size=6))
        T = fk.forward_kinematics(q)
        try:
            q_ik, info = dls.inverse_kinematics(T, q0=q0, return_info=True)
            if not info["converged"]:
                failures.append(f"no convergence: {info} for q={np.round(q, 4)}")
            if info["position_error"] > 1e-7 or info["rotation_error"] > 1e-7:
                failures.append(f"error pos={info['position_error']:.2e} "
                                f"rot={info['rotation_error']:.2e} for q={np.round(q, 4)}")
        except RuntimeError as exc:
            failures.append(f"RuntimeError({exc}) for q={np.round(q, 4)}")
    report(name, failures)


def test_damping_grows_near_singularity():
    lam0, lam_max, w_ref = 0.05, 0.5, 5e-3
    assert dls.damping(lam0, lam_max, 2 * w_ref, w_ref) == lam0
    assert dls.damping(lam0, lam_max, w_ref, w_ref) == lam0
    assert np.isclose(dls.damping(lam0, lam_max, 0.0, w_ref), lam_max)
    levels = [dls.damping(lam0, lam_max, w, w_ref)
              for w in (0.9 * w_ref, 0.5 * w_ref, 0.1 * w_ref, 0.0)]
    assert levels == sorted(levels), f"not monotone: {levels}"
    q_sing = np.array([0.4, -1.2, 1.5, 0.3, 0.0, 0.8])
    q_ok = np.array([0.4, -1.2, 1.5, 0.3, 1.2, 0.8])
    lam_sing = dls.damping(lam0, lam_max, fk.manipulability(q_sing), w_ref)
    lam_ok = dls.damping(lam0, lam_max, fk.manipulability(q_ok), w_ref)
    assert lam_sing > lam_ok, f"expected heavier damping at singularity: {lam_sing} vs {lam_ok}"


def test_dls_rejects_unreachable_target():
    R = fk.forward_kinematics(random_q())[:3, :3]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [1.0, 0.0, 0.089159]
    try:
        dls.inverse_kinematics(T)
    except RuntimeError as exc:
        print(f"  rejected unreachable target: {exc}")
        return
    raise AssertionError("expected RuntimeError for an unreachable target")


def test_dls_near_wrist_singularity_stays_bounded():
    name = "test_dls_near_wrist_singularity_stays_bounded"
    failures = []
    q_sing = np.array([0.4, -1.2, 1.5, 0.3, 0.0, 0.8])
    q_ok = np.array([0.4, -1.2, 1.5, 0.3, 1.2, 0.8])
    T_sing = fk.forward_kinematics(q_sing)
    T_ok = fk.forward_kinematics(q_ok)
    start_sing = wrap(q_sing + [0, 0, 0, 0, 0.03, 0])
    start_ok = wrap(q_ok + [0, 0, 0, 0, 0.03, 0])
    try:
        _, info_s = dls.inverse_kinematics(T_sing, q0=start_sing, return_info=True)
        _, info_o = dls.inverse_kinematics(T_ok, q0=start_ok, return_info=True)
    except RuntimeError as exc:
        failures.append(f"RuntimeError({exc})")
        report(name, failures)
        return
    for label, info in (("singular", info_s), ("regular", info_o)):
        if not np.isfinite(info["position_error"]) or not np.isfinite(info["rotation_error"]):
            failures.append(f"{label} target: non-finite error {info}")
        if not info["converged"]:
            failures.append(f"{label} target: no convergence, "
                            f"pos err {info['position_error']:.2e} m")
    if info_s["position_error"] > 1e-3:
        failures.append(f"singular target: position error {info_s['position_error']:.2e} m")
    if not info_s["lambda"] >= 0.4:
        failures.append(f"singular target: expected near-max damping, got {info_s['lambda']}")
    if not info_s["lambda"] > info_o["lambda"]:
        failures.append(f"expected higher damping near singularity: "
                        f"{info_s['lambda']} vs {info_o['lambda']}")
    print(f"  singular target: pos err {info_s['position_error']:.2e} m, "
          f"lambda {info_s['lambda']:.3f}, iters {info_s['iterations']}; "
          f"regular lambda {info_o['lambda']:.4f}")
    report(name, failures)


def test_comparison_of_both_methods():
    name = "test_comparison_of_both_methods"
    failures = []
    n = 50
    ana_dp, ana_ang, dls_dp, dls_ang = [], [], [], []
    dls_iters, dls_ok, ana_ok = [], 0, 0
    for _ in range(n):
        q = random_q()
        T = fk.forward_kinematics(q)
        try:
            q_a = ana.inverse_kinematics(T, q_prefer=q)
            dp, ang = pose_error(fk.forward_kinematics(q_a), T)
            ana_dp.append(dp)
            ana_ang.append(ang)
            ana_ok += 1
            if dp > 1e-9 or ang > 1e-9:
                failures.append(f"analytical pose error dp={dp:.2e} ang={ang:.2e} "
                                f"for q={np.round(q, 4)}")
        except ValueError as exc:
            failures.append(f"analytical ValueError({exc}) for q={np.round(q, 4)}")
        try:
            q_d, info = dls.inverse_kinematics(T, return_info=True)
            dp, ang = pose_error(fk.forward_kinematics(q_d), T)
            dls_dp.append(dp)
            dls_ang.append(ang)
            dls_iters.append(info["iterations"])
            dls_ok += 1
            if dp > 1e-6 or ang > 1e-6:
                failures.append(f"dls pose error dp={dp:.2e} ang={ang:.2e} "
                                f"for q={np.round(q, 4)}")
        except RuntimeError as exc:
            failures.append(f"dls RuntimeError({exc}) for q={np.round(q, 4)}")
    print(f"IK method comparison over {n} random targets")
    print(f"  analytical  solved {ana_ok}/{n}   max pos err "
          f"{max(ana_dp):.2e} m   max rot err {max(ana_ang):.2e} rad")
    print(f"  damped lsq  solved {dls_ok}/{n}   max pos err "
          f"{max(dls_dp):.2e} m   max rot err {max(dls_ang):.2e} rad   "
          f"mean iterations {np.mean(dls_iters):.1f}")
    report(name, failures)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"All {len(tests)} tests passed")
