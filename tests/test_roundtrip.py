"""
Comprehensive round-trip test for UR5 kinematics:
Random reachable pose -> IK (analytical and DLS) -> dh_fk.forward_kinematics.

Run from repo root:
    C:\\Users\\mansi\\anaconda3\\python.exe tests/test_roundtrip.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import dh_fk as fk
import ik_analytical as ana
import ik_dls as dls


def pose_error(T_target, T_actual):
    """Compute position error (m) and rotation error (rad)."""
    dp = float(np.linalg.norm(T_target[:3, 3] - T_actual[:3, 3]))
    dR = T_target[:3, :3] @ T_actual[:3, :3].T
    cos_t = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    v = np.array([
        dR[2, 1] - dR[1, 2],
        dR[0, 2] - dR[2, 0],
        dR[1, 0] - dR[0, 1],
    ]) / 2.0
    ang = float(np.arctan2(np.linalg.norm(v), cos_t))
    return dp, ang


def run_roundtrip_test(n_poses=50, seed=42):
    print(f"================================================================")
    print(f"  ROUND-TRIP TEST: {n_poses} RANDOM REACHABLE POSES")
    print(f"================================================================")
    rng = np.random.default_rng(seed)

    ana_pos_errors, ana_rot_errors = [], []
    dls_pos_errors, dls_rot_errors = [], []
    dls_iterations = []

    ana_failures = []
    dls_failures = []

    for i in range(n_poses):
        # Generate a known reachable pose from random joint angles
        q_true = rng.uniform(-np.pi, np.pi, size=6)
        T_target = fk.forward_kinematics(q_true)

        # 1. Analytical IK
        try:
            q_ana = ana.inverse_kinematics(T_target, q_prefer=q_true)
            T_ana = fk.forward_kinematics(q_ana)
            dp_ana, dang_ana = pose_error(T_target, T_ana)
            ana_pos_errors.append(dp_ana)
            ana_rot_errors.append(dang_ana)

            # Failure threshold: 1e-6 m position error, 1e-5 rad rotation error
            if dp_ana > 1e-6 or dang_ana > 1e-5:
                ana_failures.append({
                    "pose_idx": i,
                    "q_true": q_true,
                    "q_ana": q_ana,
                    "pos_err": dp_ana,
                    "rot_err": dang_ana,
                    "reason": "Pose error exceeded threshold",
                })
        except Exception as exc:
            ana_failures.append({
                "pose_idx": i,
                "q_true": q_true,
                "pos_err": np.nan,
                "rot_err": np.nan,
                "reason": str(exc),
            })

        # 2. Numerical DLS IK (singularity-aware)
        try:
            q_dls, info = dls.inverse_kinematics(
                T_target,
                q0=np.zeros(6),
                multi_start=True,
                return_info=True,
                raise_on_fail=False,
            )
            T_dls = fk.forward_kinematics(q_dls)
            dp_dls, dang_dls = pose_error(T_target, T_dls)
            dls_pos_errors.append(dp_dls)
            dls_rot_errors.append(dang_dls)
            dls_iterations.append(info["iterations"])

            if not info["converged"] or dp_dls > 1e-4 or dang_dls > 1e-4:
                dls_failures.append({
                    "pose_idx": i,
                    "q_true": q_true,
                    "q_dls": q_dls,
                    "pos_err": dp_dls,
                    "rot_err": dang_dls,
                    "iterations": info["iterations"],
                    "reason": "DLS did not converge within tolerance",
                })
        except Exception as exc:
            dls_failures.append({
                "pose_idx": i,
                "q_true": q_true,
                "pos_err": np.nan,
                "rot_err": np.nan,
                "reason": str(exc),
            })

    print(f"\n--- ANALYTICAL IK RESULTS ---")
    print(f"  Solved successfully : {n_poses - len(ana_failures)} / {n_poses} ({100 * (n_poses - len(ana_failures)) / n_poses:.1f}%)")
    if ana_pos_errors:
        print(f"  Mean position error : {np.mean(ana_pos_errors):.3e} m")
        print(f"  Max position error  : {np.max(ana_pos_errors):.3e} m")
        print(f"  Mean rotation error : {np.mean(ana_rot_errors):.3e} rad")
        print(f"  Max rotation error  : {np.max(ana_rot_errors):.3e} rad")

    print(f"\n--- DLS IK (SINGULARITY-AWARE) RESULTS ---")
    print(f"  Solved successfully : {n_poses - len(dls_failures)} / {n_poses} ({100 * (n_poses - len(dls_failures)) / n_poses:.1f}%)")
    if dls_pos_errors:
        print(f"  Mean position error : {np.mean(dls_pos_errors):.3e} m")
        print(f"  Max position error  : {np.max(dls_pos_errors):.3e} m")
        print(f"  Mean rotation error : {np.mean(dls_rot_errors):.3e} rad")
        print(f"  Max rotation error  : {np.max(dls_rot_errors):.3e} rad")
        print(f"  Mean iterations     : {np.mean(dls_iterations):.1f}")
        print(f"  Max iterations      : {np.max(dls_iterations)}")

    if ana_failures:
        print(f"\nANALYTICAL FAILURES ({len(ana_failures)}):")
        for f in ana_failures:
            print(f"  Pose {f['pose_idx']}: {f['reason']} (pos_err={f['pos_err']}, rot_err={f['rot_err']})")
    else:
        print(f"\n>> Analytical IK: 0 failures! All poses perfectly recovered.")

    if dls_failures:
        print(f"\nDLS FAILURES ({len(dls_failures)}):")
        for f in dls_failures:
            print(f"  Pose {f['pose_idx']}: {f['reason']} (pos_err={f['pos_err']}, rot_err={f['rot_err']})")
    else:
        print(f">> DLS IK: 0 failures! All poses successfully converged.")

    return len(ana_failures) == 0 and len(dls_failures) == 0


if __name__ == "__main__":
    success = run_roundtrip_test(n_poses=50)
    sys.exit(0 if success else 1)
