"""
M5: Generate publication-quality plots and summary comparison tables
from experimental benchmark results.

Deliverables:
- Plot 1: Success Rate Comparison (Standard vs Near-Singularity poses)
- Plot 2: Position Error Distribution (Box plot & log-scale)
- Plot 3: IK Solve Time and Full Cycle Time Comparison
- Plot 4: Numerical IK Convergence Iterations Distribution
- Plot 5: Singularity-Aware Adaptive Damping Curve (Novelty validation)
- Plot 6: 6-DOF Joint Trajectories q(t) across full Pick-and-Place Cycle
- Plot 7: Joint Velocities and Accelerations Profile (Quintic vs Cubic)
- Plot 8: Manipulability Index w(t) along Pick-and-Place Path

Author: Satwik (M5)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Styling configuration
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--"
})

COLORS = {
    "analytical": "#1f77b4",            # Blue
    "dls_fixed": "#d62728",             # Red
    "dls_singularity_aware": "#2ca02c",  # Green
}

LABELS = {
    "analytical": "Analytical IK (Closed-form)",
    "dls_fixed": "DLS (Fixed Damping λ=1e-3)",
    "dls_singularity_aware": "Singularity-Aware DLS (Novelty)",
}


def plot_success_rates(df_bench, df_sing):
    """Plot 1: Success Rate Comparison."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    methods = ["analytical", "dls_fixed", "dls_singularity_aware"]
    
    bench_rates = []
    sing_rates = []
    for m in methods:
        sub_b = df_bench[df_bench["method"] == m]
        sub_s = df_sing[df_sing["method"] == m]
        bench_rates.append((sub_b["success"].sum() / len(sub_b)) * 100.0)
        sing_rates.append((sub_s["success"].sum() / len(sub_s)) * 100.0)
        
    x = np.arange(len(methods))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, bench_rates, width, label="Standard Pick-and-Place (50 Runs)", color="#3498db", edgecolor="black")
    rects2 = ax.bar(x + width/2, sing_rates, width, label="Near-Singularity Stress Test (30 Poses)", color="#e67e22", edgecolor="black")
    
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("IK Method Success Rate: Standard vs Near-Singularity Poses")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in methods], rotation=10, ha="right")
    ax.set_ylim(0, 115)
    ax.axhline(100, color="gray", linestyle=":", alpha=0.7)
    ax.legend(loc="lower right")
    ax.grid(True, axis="y")
    
    # Add labels on bars
    for rects in (rects1, rects2):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontweight="bold", fontsize=9)
            
    out_path = os.path.join(PLOTS_DIR, "plot1_success_rate.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_position_error_distribution(df_bench):
    """Plot 2: Position Error Distribution (Box plot & Violin on log scale)."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    methods = ["analytical", "dls_fixed", "dls_singularity_aware"]
    
    data = []
    for m in methods:
        sub = df_bench[(df_bench["method"] == m) & (df_bench["success"])]
        # Clip minimum to 1e-12 mm for log scale representation
        errs = np.clip(sub["max_pos_error_mm"].values, 1e-12, None)
        data.append(errs)
        
    box = ax.boxplot(data, tick_labels=[LABELS[m] for m in methods], patch_artist=True, showmeans=True,
                     meanprops={"marker": "D", "markerfacecolor": "yellow", "markeredgecolor": "black"})
    
    colors = ["#a8d5e5", "#f7b7b7", "#b7e4c7"]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        
    ax.set_yscale("log")
    ax.set_ylabel("Maximum Position Error (mm) [Log Scale]")
    ax.set_title("End-Effector Position Error Distribution Across 50 Trials")
    ax.grid(True, which="both", axis="y")
    
    # Add threshold reference line at 1 mm and 0.1 mm
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, label="1.0 mm Industrial Tolerance")
    ax.axhline(0.01, color="blue", linestyle=":", alpha=0.7, label="0.01 mm High Precision")
    ax.legend(loc="upper left")
    
    out_path = os.path.join(PLOTS_DIR, "plot2_position_error.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_solve_and_cycle_time(df_bench):
    """Plot 3: Solve Time and Total Cycle Time Comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    methods = ["analytical", "dls_fixed", "dls_singularity_aware"]
    x = np.arange(len(methods))
    
    # Solve time per pose (ms)
    solve_times = [df_bench[df_bench["method"] == m]["solve_time_per_pose_ms"].mean() for m in methods]
    solve_std = [df_bench[df_bench["method"] == m]["solve_time_per_pose_ms"].std() for m in methods]
    
    # Cycle times (s)
    cycle_times = [df_bench[(df_bench["method"] == m) & (df_bench["success"])]["cycle_time_s"].mean() for m in methods]
    cycle_std = [df_bench[(df_bench["method"] == m) & (df_bench["success"])]["cycle_time_s"].std() for m in methods]
    
    bar_colors = ["#1f77b4", "#d62728", "#2ca02c"]
    
    # Panel 1: Solve Time
    rects1 = ax1.bar(x, solve_times, yerr=solve_std, capsize=5, color=bar_colors, edgecolor="black", alpha=0.85)
    ax1.set_ylabel("Mean Computation Time per Pose (ms)")
    ax1.set_title("IK Solver Computational Latency")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Analytical", "Fixed DLS", "Adaptive DLS\n(Novelty)"])
    ax1.grid(True, axis="y")
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f} ms", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
        
    # Panel 2: Full Pick-and-Place Cycle Time
    rects2 = ax2.bar(x, cycle_times, yerr=cycle_std, capsize=5, color=bar_colors, edgecolor="black", alpha=0.85)
    ax2.set_ylabel("Total Cycle Time (s)")
    ax2.set_title("Full Pick-and-Place Cycle Execution Time")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Analytical", "Fixed DLS", "Adaptive DLS\n(Novelty)"])
    ax2.set_ylim(0, max(cycle_times) * 1.25)
    ax2.grid(True, axis="y")
    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f"{h:.2f} s", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
        
    out_path = os.path.join(PLOTS_DIR, "plot3_solve_and_cycle_time.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_iterations_distribution(df_bench):
    """Plot 4: Numerical IK Convergence Iterations Histogram."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    fixed_iters = df_bench[df_bench["method"] == "dls_fixed"]["total_iterations"] / 6.0
    adapt_iters = df_bench[df_bench["method"] == "dls_singularity_aware"]["total_iterations"] / 6.0
    
    bins = np.linspace(1, max(max(fixed_iters), max(adapt_iters)) + 5, 20)
    ax.hist(fixed_iters, bins=bins, alpha=0.6, label="Fixed DLS (λ=1e-3)", color="red", edgecolor="black")
    ax.hist(adapt_iters, bins=bins, alpha=0.6, label="Adaptive DLS (Singularity-Aware)", color="green", edgecolor="black")
    
    ax.set_xlabel("Iterations per Target Pose")
    ax.set_ylabel("Frequency (Number of Poses)")
    ax.set_title("Distribution of Convergence Iterations for Numerical Methods")
    ax.legend(loc="upper right")
    ax.grid(True)
    
    out_path = os.path.join(PLOTS_DIR, "plot4_convergence_iterations.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_singularity_damping_curve():
    """Plot 5: Novel Singularity-Aware Damping Profile vs Manipulability."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    lam0 = 1e-3
    lam_max = 0.5
    w_ref = 1e-3
    
    w_vals = np.linspace(0.0, 3e-3, 500)
    lam_vals = []
    for w in w_vals:
        if w >= w_ref:
            lam_vals.append(lam0)
        else:
            gap = 1.0 - w / w_ref
            lam_vals.append(lam0 + (lam_max - lam0) * (gap ** 2))
            
    ax.plot(w_vals * 1000.0, lam_vals, color="#2ca02c", linewidth=2.5, label=r"Adaptive Damping $\lambda(w)$")
    ax.axhline(lam0, color="#d62728", linestyle="--", linewidth=1.8, label=r"Constant Damping $\lambda_0 = 10^{-3}$")
    ax.axvline(w_ref * 1000.0, color="navy", linestyle=":", linewidth=1.5, label=r"Threshold $w_{ref} = 1.0 \times 10^{-3}$")
    
    ax.fill_between(w_vals * 1000.0, lam_vals, lam0, color="#2ca02c", alpha=0.15, label="Singularity Protection Buffer")
    
    ax.set_xlabel(r"Yoshikawa Manipulability Measure $w \times 10^{-3}$")
    ax.set_ylabel(r"Damping Factor $\lambda$")
    ax.set_title(r"Novelty Verification: Singularity-Aware Damping Adaptation Curve")
    ax.set_ylim(-0.02, 0.55)
    ax.legend(loc="upper right")
    ax.grid(True)
    
    # Annotations
    ax.annotate("Near-Singular Zone\n(High Damping, Prevents Spikes)",
                xy=(0.2, 0.4), xytext=(0.6, 0.45),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
                fontweight="bold")
    ax.annotate("Well-Conditioned Zone\n(Minimal Damping, Fast Convergence)",
                xy=(1.8, 0.02), xytext=(1.4, 0.15),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6))
                
    out_path = os.path.join(PLOTS_DIR, "plot5_singularity_damping_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_joint_trajectories(df_traj):
    """Plot 6: 6-DOF Joint Trajectory Profiles across Full Pick-and-Place Cycle."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True, dpi=300)
    fig.suptitle("6-DOF UR5 Joint Space Trajectories $q_1(t) \\dots q_6(t)$ for Full Pick-and-Place Cycle", fontsize=14, y=0.99)
    
    joint_names = [
        "Joint 1: Shoulder Pan", "Joint 2: Shoulder Lift",
        "Joint 3: Elbow", "Joint 4: Wrist 1",
        "Joint 5: Wrist 2", "Joint 6: Wrist 3"
    ]
    
    t = df_traj["time_s"].values
    
    for j in range(6):
        ax = axes[j // 2, j % 2]
        q_deg = np.rad2deg(df_traj[f"q{j+1}_rad"].values)
        ax.plot(t, q_deg, color="#1f77b4", linewidth=2.0)
        ax.set_ylabel(f"{joint_names[j]}\nAngle (deg)")
        ax.grid(True)
        
        # Add phase markers
        # Phases roughly at: 1.5, 2.3, 3.1, 5.1, 5.9, 6.7, 8.2 s
        phase_times = [1.5, 2.3, 3.1, 5.1, 5.9, 6.7, 8.2]
        for pt in phase_times:
            ax.axvline(pt, color="gray", linestyle=":", alpha=0.5)
            
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 1].set_xlabel("Time (s)")
    
    out_path = os.path.join(PLOTS_DIR, "plot6_joint_trajectories_6dof.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_joint_velocities_and_accelerations(df_traj):
    """Plot 7: Velocities and Accelerations comparing Quintic vs Cubic."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, dpi=300)
    t = df_traj["time_s"].values
    
    # Plot Joint 2 (Shoulder lift) velocities & accelerations
    ax1.plot(t, df_traj["qd2_rad_s"], label=r"Quintic Polynomial (Smooth $\ddot{q}=0$)", color="#2ca02c", linewidth=2.2)
    ax1.plot(t, df_traj["qd2_cubic_rad_s"], label="Cubic Polynomial", color="#e67e22", linestyle="--", linewidth=1.8)
    ax1.set_ylabel("Joint 2 Velocity (rad/s)")
    ax1.set_title("Kinematic Trajectory Comparison: Quintic vs Cubic Polynomials (Joint 2)")
    ax1.legend(loc="upper right")
    ax1.grid(True)
    
    ax2.plot(t, df_traj["qdd2_rad_s2"], label="Quintic Acceleration (Continuous at Boundaries)", color="#2ca02c", linewidth=2.2)
    ax2.plot(t, df_traj["qdd2_cubic_rad_s2"], label="Cubic Acceleration (Discontinuous Step Jumps)", color="#e67e22", linestyle="--", linewidth=1.8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Joint 2 Acceleration (rad/s²)")
    ax2.legend(loc="upper right")
    ax2.grid(True)
    
    out_path = os.path.join(PLOTS_DIR, "plot7_joint_velocities_accelerations.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_manipulability_profile(df_traj):
    """Plot 8: End-Effector Trajectory & Manipulability."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, dpi=300)
    t = df_traj["time_s"].values
    
    # EE Positions
    ax1.plot(t, df_traj["ee_x"] * 1000.0, label="X position (mm)", color="#1f77b4")
    ax1.plot(t, df_traj["ee_y"] * 1000.0, label="Y position (mm)", color="#ff7f0e")
    ax1.plot(t, df_traj["ee_z"] * 1000.0, label="Z position (mm)", color="#2ca02c")
    ax1.set_ylabel("End-Effector Pos (mm)")
    ax1.set_title("Cartesian End-Effector Tracking and Manipulability Index")
    ax1.legend(loc="upper right")
    ax1.grid(True)
    
    # Manipulability
    ax2.plot(t, df_traj["manipulability"], color="purple", linewidth=2.0, label=r"Yoshikawa Manipulability $w = \sqrt{\det(J J^T)}$")
    ax2.axhline(1e-3, color="red", linestyle="--", label=r"Singularity Threshold $w_{ref} = 10^{-3}$")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Manipulability Index $w$")
    ax2.legend(loc="lower right")
    ax2.grid(True)
    
    out_path = os.path.join(PLOTS_DIR, "plot8_manipulability_profile.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    bench_csv = os.path.join(RESULTS_DIR, "benchmark_50_runs.csv")
    sing_csv = os.path.join(RESULTS_DIR, "singularity_stress_test.csv")
    traj_csv = os.path.join(RESULTS_DIR, "trajectory_sample_timeseries.csv")
    
    if not (os.path.exists(bench_csv) and os.path.exists(sing_csv) and os.path.exists(traj_csv)):
        print("Error: CSV result files not found! Please run 'experiments/run_experiments.py' first.")
        sys.exit(1)
        
    print("Loading CSV results from results/...")
    df_bench = pd.read_csv(bench_csv)
    df_sing = pd.read_csv(sing_csv)
    df_traj = pd.read_csv(traj_csv)
    
    print("\nGenerating publication-quality plots...")
    plot_success_rates(df_bench, df_sing)
    plot_position_error_distribution(df_bench)
    plot_solve_and_cycle_time(df_bench)
    plot_iterations_distribution(df_bench)
    plot_singularity_damping_curve()
    plot_joint_trajectories(df_traj)
    plot_joint_velocities_and_accelerations(df_traj)
    plot_manipulability_profile(df_traj)
    
    print(f"\nAll 8 plots successfully generated and saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
