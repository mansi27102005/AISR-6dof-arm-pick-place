"""
M5: Run experimental evaluation and benchmarks comparing:
1. Analytical Inverse Kinematics (closed-form)
2. Numerical DLS Inverse Kinematics (fixed damping)
3. Singularity-Aware DLS Inverse Kinematics (adaptive damping - novelty)

Also generates full pick-and-place trajectories and logs all CSV data to results/.
Author: Satwik (M5)
"""
import os
import sys
import time
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import dh_fk as fk
import ik_analytical as ana
import ik_dls as dls
import trajectory as traj

# Ensure reproducibility
RNG = np.random.default_rng(42)

# Output directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def rot_downward(yaw=0.0):
    """
    Generate rotation matrix for gripper pointing downwards (z_flange along -z_base).
    Yaw rotates about z_base.
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([
        [cy, -sy, 0.0],
        [sy,  cy, 0.0],
        [0.0, 0.0, 1.0]
    ])
    # Downward pointing: Rx(pi)
    Rx_pi = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ])
    return Rz @ Rx_pi


def make_pose(position, orientation_matrix):
    """Create 4x4 homogeneous transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = orientation_matrix
    T[:3, 3] = position
    return T


def pose_error(T_target, T_actual):
    """Compute position error (m) and rotation error (rad)."""
    dp = float(np.linalg.norm(T_target[:3, 3] - T_actual[:3, 3]))
    dR = T_target[:3, :3] @ T_actual[:3, :3].T
    cos_t = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    v = np.array([
        dR[2, 1] - dR[1, 2],
        dR[0, 2] - dR[2, 0],
        dR[1, 0] - dR[0, 1]
    ]) / 2.0
    ang = float(np.arctan2(np.linalg.norm(v), cos_t))
    return dp, ang


def solve_ik(method, T_target, q_prev=None):
    """
    Solve IK using one of the three methods:
    'analytical', 'dls_fixed', 'dls_singularity_aware'
    Returns: (q_sol, success, solve_time_ms, pos_err_mm, rot_err_rad, iterations, info)
    """
    t0 = time.perf_counter()
    q_sol = None
    success = False
    iterations = 1
    info = {}
    
    if method == "analytical":
        try:
            q_sol = ana.inverse_kinematics(T_target, q_prefer=q_prev)
            success = True
            iterations = 1
        except Exception as e:
            success = False
            q_sol = q_prev if q_prev is not None else np.zeros(6)
            info["error"] = str(e)
            
    elif method == "dls_fixed":
        try:
            # Fixed damping: lam_max = lam0 = 1e-3, single start or multi-start
            q_sol, dls_info = dls.inverse_kinematics(
                T_target, q0=q_prev,
                lam0=1e-3, lam_max=1e-3,
                max_iter=150, pos_tol=1e-5, rot_tol=1e-5,
                multi_start=False, return_info=True, raise_on_fail=False
            )
            success = dls_info["converged"]
            iterations = dls_info["iterations"]
            info = dls_info
        except Exception as e:
            success = False
            q_sol = q_prev if q_prev is not None else np.zeros(6)
            info["error"] = str(e)
            
    elif method == "dls_singularity_aware":
        try:
            # Adaptive damping: lam0=1e-3, lam_max=0.5, w_ref=1e-3
            q_sol, dls_info = dls.inverse_kinematics(
                T_target, q0=q_prev,
                lam0=1e-3, lam_max=0.5, w_ref=1e-3,
                max_iter=150, pos_tol=1e-5, rot_tol=1e-5,
                multi_start=True, return_info=True, raise_on_fail=False
            )
            success = dls_info["converged"]
            iterations = dls_info["iterations"]
            info = dls_info
        except Exception as e:
            success = False
            q_sol = q_prev if q_prev is not None else np.zeros(6)
            info["error"] = str(e)
    else:
        raise ValueError(f"Unknown method {method}")
        
    t1 = time.perf_counter()
    solve_time_ms = (t1 - t0) * 1000.0
    
    if success and q_sol is not None:
        T_act = fk.forward_kinematics(q_sol)
        dp, dang = pose_error(T_target, T_act)
        pos_err_mm = dp * 1000.0
        rot_err_rad = dang
    else:
        pos_err_mm = float(info.get("position_error", 999.0)) * 1000.0 if "position_error" in info else 999.0
        rot_err_rad = float(info.get("rotation_error", 999.0)) if "rotation_error" in info else 999.0

    return q_sol, success, solve_time_ms, pos_err_mm, rot_err_rad, iterations, info


def sample_reachable_pick_positions(n=50):
    """
    Sample n random object positions across the UR5 table pick zone.
    Returns array of shape (n, 3).
    """
    # Table bounds in base frame:
    # Radius r in [0.35, 0.65], azimuth phi in [-pi/3, pi/3], z in [0.05, 0.25]
    radii = RNG.uniform(0.35, 0.65, size=n)
    phis = RNG.uniform(-np.pi / 3.0, np.pi / 3.0, size=n)
    zs = RNG.uniform(0.05, 0.22, size=n)
    
    xs = radii * np.cos(phis)
    ys = radii * np.sin(phis)
    return np.column_stack([xs, ys, zs])


def run_50_benchmark_trials(num_trials=50):
    """
    Run 50 pick-and-place trials comparing Analytical, DLS (fixed), and DLS (Singularity-Aware).
    Each trial executes the 6 canonical phases:
    1. Standby
    2. Approach Pick (z + 0.10)
    3. Grasp Pick (z)
    4. Lift (z + 0.15)
    5. Approach Place (z_drop + 0.15)
    6. Place (z_drop)
    7. Retreat to Standby
    """
    print(f"\n=======================================================")
    print(f"  RUNNING BENCHMARK: {num_trials} RANDOM OBJECT POSITIONS")
    print(f"=======================================================")
    
    pick_positions = sample_reachable_pick_positions(num_trials)
    place_position = np.array([-0.35, 0.35, 0.08])
    standby_q = np.array([0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0])
    
    methods = ["analytical", "dls_fixed", "dls_singularity_aware"]
    results = []
    
    # Timing parameters per segment (seconds)
    segment_durations = [1.5, 0.8, 0.8, 2.0, 0.8, 1.5]
    total_trajectory_nominal_time = sum(segment_durations)
    
    for trial_idx in range(num_trials):
        pick_pos = pick_positions[trial_idx]
        R_pick = rot_downward(yaw=np.arctan2(pick_pos[1], pick_pos[0]))
        R_place = rot_downward(yaw=np.arctan2(place_position[1], place_position[0]))
        
        # 6 Target poses
        target_poses = [
            make_pose(pick_pos + np.array([0, 0, 0.10]), R_pick),   # Approach
            make_pose(pick_pos, R_pick),                           # Grasp
            make_pose(pick_pos + np.array([0, 0, 0.15]), R_pick),   # Lift
            make_pose(place_position + np.array([0, 0, 0.15]), R_place), # Approach Place
            make_pose(place_position, R_place),                    # Place
            make_pose(place_position + np.array([0, 0, 0.15]), R_place), # Retreat
        ]
        
        for method in methods:
            q_current = standby_q.copy()
            waypoints = [q_current]
            all_success = True
            total_solve_time = 0.0
            max_pos_err = 0.0
            max_rot_err = 0.0
            total_iters = 0
            
            for phase_idx, T_target in enumerate(target_poses):
                q_sol, succ, s_time, p_err, r_err, iters, info = solve_ik(method, T_target, q_prev=q_current)
                total_solve_time += s_time
                total_iters += iters
                max_pos_err = max(max_pos_err, p_err)
                max_rot_err = max(max_rot_err, r_err)
                
                if not succ or p_err > 5.0:  # > 5 mm threshold
                    all_success = False
                    break
                else:
                    waypoints.append(q_sol)
                    q_current = q_sol
                    
            # If all phases succeeded, evaluate trajectory dynamics
            if all_success and len(waypoints) == len(target_poses) + 1:
                times, q_tr, qd_tr, qdd_tr = traj.plan_multisegment_trajectory(
                    waypoints, segment_durations, dt=0.02, method="quintic"
                )
                max_joint_vel = float(np.max(np.abs(qd_tr)))
                max_joint_acc = float(np.max(np.abs(qdd_tr)))
                # Manipulability along trajectory
                manip_vals = [fk.manipulability(q) for q in q_tr[::10]]
                min_manip = float(np.min(manip_vals))
                cycle_time = float(times[-1]) + (total_solve_time / 1000.0)
            else:
                max_joint_vel = np.nan
                max_joint_acc = np.nan
                min_manip = np.nan
                cycle_time = np.nan
                
            results.append({
                "trial_id": trial_idx + 1,
                "method": method,
                "pick_x": round(pick_pos[0], 4),
                "pick_y": round(pick_pos[1], 4),
                "pick_z": round(pick_pos[2], 4),
                "success": bool(all_success),
                "solve_time_total_ms": round(total_solve_time, 3),
                "solve_time_per_pose_ms": round(total_solve_time / len(target_poses), 3),
                "max_pos_error_mm": round(max_pos_err, 6),
                "max_rot_error_rad": round(max_rot_err, 6),
                "total_iterations": total_iters,
                "cycle_time_s": round(cycle_time, 3) if not np.isnan(cycle_time) else np.nan,
                "max_joint_vel_rad_s": round(max_joint_vel, 4) if not np.isnan(max_joint_vel) else np.nan,
                "max_joint_acc_rad_s2": round(max_joint_acc, 4) if not np.isnan(max_joint_acc) else np.nan,
                "min_manipulability": round(min_manip, 6) if not np.isnan(min_manip) else np.nan,
            })
            
        if (trial_idx + 1) % 10 == 0:
            print(f"  Completed {trial_idx + 1}/{num_trials} benchmark trials...")
            
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "benchmark_50_runs.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"  [SAVED] Benchmark results saved to: {csv_path}")
    return df_results


def run_singularity_stress_test(num_poses_per_type=10):
    """
    Stress test specifically designed to prove the NOVELTY CLAIMS:
    Tests configurations directly adjacent to UR5 singular surfaces:
    1. Wrist Singularity: Joint 5 -> 0 (axes of joints 4 and 6 become collinear)
    2. Shoulder Singularity: wrist centre aligns with joint 1 axis (r -> d4)
    3. Elbow Singularity: arm fully outstretched (joint 3 -> 0)
    """
    print(f"\n=======================================================")
    print(f"  RUNNING SINGULARITY STRESS TEST ({num_poses_per_type * 3} POSES)")
    print(f"=======================================================")
    
    stress_cases = []
    
    # 1. Wrist singularities (q5 in [0.0001, 0.03] rad)
    for i in range(num_poses_per_type):
        q5_val = 0.0001 + (i / num_poses_per_type) * 0.02
        q_sing = np.array([0.4, -1.2, 1.5, 0.3, q5_val, 0.8])
        T_target = fk.forward_kinematics(q_sing)
        stress_cases.append(("wrist", T_target, q_sing, fk.manipulability(q_sing)))
        
    # 2. Shoulder singularities (near base z-axis)
    for i in range(num_poses_per_type):
        d4 = fk.DH_TABLE[3, 2]
        r_offset = d4 + 0.002 * (i + 1)  # Just outside shoulder singular cylinder
        ang = RNG.uniform(-np.pi, np.pi)
        p = np.array([r_offset * np.cos(ang), r_offset * np.sin(ang), 0.35])
        T_target = make_pose(p, rot_downward(ang))
        stress_cases.append(("shoulder", T_target, None, 1e-4 * (i + 1)))
        
    # 3. Elbow singularities (extended reach r -> a2 + a3)
    for i in range(num_poses_per_type):
        max_reach = abs(fk.DH_TABLE[1, 0]) + abs(fk.DH_TABLE[2, 0])
        r_reach = max_reach - 0.005 * (i + 1)
        ang = RNG.uniform(-np.pi/4, np.pi/4)
        p = np.array([r_reach * np.cos(ang), r_reach * np.sin(ang), 0.15])
        T_target = make_pose(p, rot_downward(ang))
        stress_cases.append(("elbow", T_target, None, 2e-4 * (i + 1)))
        
    methods = ["analytical", "dls_fixed", "dls_singularity_aware"]
    results = []
    
    for case_id, (sing_type, T_target, q_true, w_nominal) in enumerate(stress_cases):
        q_start = np.zeros(6) if q_true is None else (q_true + RNG.uniform(-0.05, 0.05, size=6))
        
        for method in methods:
            t0 = time.perf_counter()
            if method == "analytical":
                try:
                    q_sol = ana.inverse_kinematics(T_target, q_prefer=q_start)
                    succ = True
                    iters = 1
                    lam_used = 0.0
                    w_actual = fk.manipulability(q_sol)
                except Exception as e:
                    succ = False
                    q_sol = q_start
                    iters = 1
                    lam_used = 0.0
                    w_actual = 0.0
            elif method == "dls_fixed":
                q_sol, info = dls.inverse_kinematics(
                    T_target, q0=q_start,
                    lam0=1e-3, lam_max=1e-3,
                    max_iter=250, pos_tol=1e-4, rot_tol=1e-4,
                    multi_start=False, return_info=True, raise_on_fail=False
                )
                succ = info["converged"]
                iters = info["iterations"]
                lam_used = 1e-3
                w_actual = info["manipulability"]
            elif method == "dls_singularity_aware":
                q_sol, info = dls.inverse_kinematics(
                    T_target, q0=q_start,
                    lam0=1e-3, lam_max=0.5, w_ref=1e-3,
                    max_iter=250, pos_tol=1e-4, rot_tol=1e-4,
                    multi_start=True, return_info=True, raise_on_fail=False
                )
                succ = info["converged"]
                iters = info["iterations"]
                lam_used = info["lambda"]
                w_actual = info["manipulability"]
                
            t1 = time.perf_counter()
            solve_ms = (t1 - t0) * 1000.0
            
            T_act = fk.forward_kinematics(q_sol)
            dp, dang = pose_error(T_target, T_act)
            delta_q_norm = float(np.linalg.norm(q_sol - q_start))
            
            results.append({
                "case_id": case_id + 1,
                "singularity_type": sing_type,
                "method": method,
                "manipulability_nominal": round(w_nominal, 6),
                "manipulability_actual": round(w_actual, 6),
                "damping_lambda": round(lam_used, 5),
                "success": bool(succ),
                "pos_error_mm": round(dp * 1000.0, 4),
                "rot_error_rad": round(dang, 4),
                "iterations": iters,
                "solve_time_ms": round(solve_ms, 3),
                "joint_displacement_norm": round(delta_q_norm, 4)
            })
            
    df_sing = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "singularity_stress_test.csv")
    df_sing.to_csv(csv_path, index=False)
    print(f"  [SAVED] Singularity stress test saved to: {csv_path}")
    return df_sing


def generate_time_series_trajectory_log():
    """
    Log full time-series joint trajectories for an entire pick-and-place cycle.
    Provides detailed data for plotting q(t), qdot(t), qddot(t), w(t).
    """
    print(f"\n=======================================================")
    print(f"  GENERATING TIME-SERIES TRAJECTORY LOG")
    print(f"=======================================================")
    
    standby_q = np.array([0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0])
    pick_pos = np.array([0.45, 0.20, 0.12])
    place_pos = np.array([-0.35, 0.35, 0.08])
    
    R_pick = rot_downward(np.arctan2(pick_pos[1], pick_pos[0]))
    R_place = rot_downward(np.arctan2(place_pos[1], place_pos[0]))
    
    target_poses = [
        make_pose(pick_pos + np.array([0, 0, 0.10]), R_pick),
        make_pose(pick_pos, R_pick),
        make_pose(pick_pos + np.array([0, 0, 0.15]), R_pick),
        make_pose(place_pos + np.array([0, 0, 0.15]), R_place),
        make_pose(place_pos, R_place),
        make_pose(place_pos + np.array([0, 0, 0.15]), R_place),
        fk.forward_kinematics(standby_q)
    ]
    
    waypoints = [standby_q]
    q_curr = standby_q.copy()
    for T in target_poses:
        q_sol = ana.inverse_kinematics(T, q_prefer=q_curr)
        waypoints.append(q_sol)
        q_curr = q_sol
        
    durations = [1.5, 0.8, 0.8, 2.0, 0.8, 0.8, 1.5]
    
    # 1. Quintic trajectory
    t_q, q_q, qd_q, qdd_q = traj.plan_multisegment_trajectory(waypoints, durations, dt=0.01, method="quintic")
    # 2. Cubic trajectory
    t_c, q_c, qd_c, qdd_c = traj.plan_multisegment_trajectory(waypoints, durations, dt=0.01, method="cubic")
    
    rows = []
    for i, t in enumerate(t_q):
        manip = fk.manipulability(q_q[i])
        # Position of end effector
        T_ee = fk.forward_kinematics(q_q[i])
        p_ee = T_ee[:3, 3]
        
        row = {
            "time_s": round(t, 3),
            "ee_x": round(p_ee[0], 4),
            "ee_y": round(p_ee[1], 4),
            "ee_z": round(p_ee[2], 4),
            "manipulability": round(manip, 6),
            "damping_lambda": round(dls.damping(1e-3, 0.5, manip, 1e-3), 5),
        }
        for j in range(6):
            row[f"q{j+1}_rad"] = round(q_q[i, j], 5)
            row[f"qd{j+1}_rad_s"] = round(qd_q[i, j], 5)
            row[f"qdd{j+1}_rad_s2"] = round(qdd_q[i, j], 5)
            # Add cubic velocities for comparison
            row[f"qd{j+1}_cubic_rad_s"] = round(qd_c[i, j], 5)
            row[f"qdd{j+1}_cubic_rad_s2"] = round(qdd_c[i, j], 5)
        rows.append(row)
        
    df_traj = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "trajectory_sample_timeseries.csv")
    df_traj.to_csv(csv_path, index=False)
    print(f"  [SAVED] Trajectory time-series saved to: {csv_path}")
    return df_traj


def generate_summary_table(df_bench, df_sing):
    """
    Produce the definitive comparison table required for Section 9, 12, and the slides.
    """
    print(f"\n=======================================================")
    print(f"  GENERATING COMPARISON TABLE FOR NOVELTY CLAIMS")
    print(f"=======================================================")
    
    summary = []
    for method in ["analytical", "dls_fixed", "dls_singularity_aware"]:
        sub_b = df_bench[df_bench["method"] == method]
        sub_s = df_sing[df_sing["method"] == method]
        
        success_rate_bench = (sub_b["success"].sum() / len(sub_b)) * 100.0
        success_rate_sing = (sub_s["success"].sum() / len(sub_s)) * 100.0
        
        mean_pos_err = sub_b.loc[sub_b["success"], "max_pos_error_mm"].mean()
        mean_solve_ms = sub_b["solve_time_per_pose_ms"].mean()
        mean_iters = sub_b["total_iterations"].mean() / 6.0
        mean_cycle_s = sub_b.loc[sub_b["success"], "cycle_time_s"].mean()
        mean_max_vel = sub_b.loc[sub_b["success"], "max_joint_vel_rad_s"].mean()
        
        # Singularity metrics
        sing_pos_err = sub_s.loc[sub_s["success"], "pos_error_mm"].mean()
        sing_max_jump = sub_s["joint_displacement_norm"].max()
        
        summary.append({
            "Method": method,
            "Standard Success Rate (%)": round(success_rate_bench, 1),
            "Singularity Success Rate (%)": round(success_rate_sing, 1),
            "Mean Pos Error (mm)": round(mean_pos_err, 4),
            "Singularity Pos Error (mm)": round(sing_pos_err, 4),
            "Mean Solve Time (ms)": round(mean_solve_ms, 3),
            "Mean Iterations / Pose": round(mean_iters, 1),
            "Mean Cycle Time (s)": round(mean_cycle_s, 2),
            "Max Joint Velocity (rad/s)": round(mean_max_vel, 3),
            "Max Singularity Jump Norm (rad)": round(sing_max_jump, 3),
        })
        
    df_sum = pd.DataFrame(summary)
    csv_path = os.path.join(RESULTS_DIR, "method_summary_table.csv")
    df_sum.to_csv(csv_path, index=False)
    print(f"  [SAVED] Summary table saved to: {csv_path}")
    print("\n" + df_sum.to_string(index=False))
    return df_sum


def main():
    print("Starting M5 Experiments Suite...")
    df_bench = run_50_benchmark_trials(num_trials=50)
    df_sing = run_singularity_stress_test(num_poses_per_type=10)
    df_traj = generate_time_series_trajectory_log()
    df_sum = generate_summary_table(df_bench, df_sing)
    print("\nAll experiments completed successfully! Results logged in 'results/' directory.")


if __name__ == "__main__":
    main()
