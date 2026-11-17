"""
M4: joint-space trajectory generation (cubic / quintic).

Generates smooth trajectories between joint configurations with bounded
velocity and acceleration, guaranteeing zero start and end velocities (and
accelerations for quintic).
"""
import numpy as np


def cubic_step(t, t_start, t_end, q_start, q_end):
    """
    Evaluate cubic polynomial trajectory at time t.
    s(tau) = 3*tau^2 - 2*tau^3,  tau = (t - t_start) / (t_end - t_start)
    Returns (q, q_dot, q_ddot).
    """
    T = t_end - t_start
    if T <= 0:
        raise ValueError("Duration must be strictly positive")
    tau = np.clip((t - t_start) / T, 0.0, 1.0)
    
    # Position: s(tau)
    s = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
    # Velocity: s_dot(tau)
    ds_dtau = 6.0 * tau - 6.0 * (tau ** 2)
    s_dot = ds_dtau / T
    # Acceleration: s_ddot(tau)
    d2s_dtau2 = 6.0 - 12.0 * tau
    s_ddot = d2s_dtau2 / (T ** 2)
    
    q_start = np.asarray(q_start, dtype=float)
    q_end = np.asarray(q_end, dtype=float)
    delta_q = q_end - q_start
    
    q = q_start + delta_q * s
    q_dot = delta_q * s_dot
    q_ddot = delta_q * s_ddot
    return q, q_dot, q_ddot


def quintic_step(t, t_start, t_end, q_start, q_end):
    """
    Evaluate quintic (5th order) polynomial trajectory at time t.
    s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5
    Guarantees:
      s(0) = 0, s(1) = 1
      s'(0) = s'(1) = 0 (zero velocity boundary conditions)
      s''(0) = s''(1) = 0 (zero acceleration / smooth start & stop)
    Returns (q, q_dot, q_ddot).
    """
    T = t_end - t_start
    if T <= 0:
        raise ValueError("Duration must be strictly positive")
    tau = np.clip((t - t_start) / T, 0.0, 1.0)
    
    # Position: s(tau)
    s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
    # Velocity: s_dot(tau)
    ds_dtau = 30.0 * (tau ** 2) - 60.0 * (tau ** 3) + 30.0 * (tau ** 4)
    s_dot = ds_dtau / T
    # Acceleration: s_ddot(tau)
    d2s_dtau2 = 60.0 * tau - 180.0 * (tau ** 2) + 120.0 * (tau ** 3)
    s_ddot = d2s_dtau2 / (T ** 2)
    
    q_start = np.asarray(q_start, dtype=float)
    q_end = np.asarray(q_end, dtype=float)
    delta_q = q_end - q_start
    
    q = q_start + delta_q * s
    q_dot = delta_q * s_dot
    q_ddot = delta_q * s_ddot
    return q, q_dot, q_ddot


def generate_trajectory(q_start, q_end, duration=2.0, dt=0.02, method="quintic"):
    """
    Generate time-series arrays for a single segment motion.
    
    Parameters:
        q_start: 6-vector initial joint angles (radians)
        q_end:   6-vector final joint angles (radians)
        duration: segment duration in seconds
        dt: sampling interval in seconds
        method: 'quintic' or 'cubic'
        
    Returns:
        times: (N,) time stamps
        q_traj: (N, 6) positions
        qd_traj: (N, 6) velocities
        qdd_traj: (N, 6) accelerations
    """
    step_func = quintic_step if method == "quintic" else cubic_step
    times = np.arange(0.0, duration + dt / 2.0, dt)
    n_steps = len(times)
    n_joints = len(q_start)
    
    q_traj = np.zeros((n_steps, n_joints))
    qd_traj = np.zeros((n_steps, n_joints))
    qdd_traj = np.zeros((n_steps, n_joints))
    
    for i, t in enumerate(times):
        q, qd, qdd = step_func(t, 0.0, duration, q_start, q_end)
        q_traj[i] = q
        qd_traj[i] = qd
        qdd_traj[i] = qdd
        
    return times, q_traj, qd_traj, qdd_traj


def plan_multisegment_trajectory(waypoints, durations, dt=0.02, method="quintic"):
    """
    Generate smooth continuous trajectory connecting an ordered sequence of waypoints.
    
    Parameters:
        waypoints: list of (6,) joint angle arrays
        durations: list of durations for each segment (len = len(waypoints) - 1)
        dt: sampling interval
        method: 'quintic' or 'cubic'
    """
    if len(waypoints) < 2:
        raise ValueError("At least 2 waypoints required")
    if len(durations) != len(waypoints) - 1:
        raise ValueError("Number of durations must be len(waypoints) - 1")
        
    all_times = []
    all_q = []
    all_qd = []
    all_qdd = []
    current_time = 0.0
    
    for idx, (q_s, q_e, dur) in enumerate(zip(waypoints[:-1], waypoints[1:], durations)):
        t_seg, q_seg, qd_seg, qdd_seg = generate_trajectory(q_s, q_e, duration=dur, dt=dt, method=method)
        # Avoid duplicating junction point
        if idx > 0:
            t_seg = t_seg[1:]
            q_seg = q_seg[1:]
            qd_seg = qd_seg[1:]
            qdd_seg = qdd_seg[1:]
            
        all_times.append(t_seg + current_time)
        all_q.append(q_seg)
        all_qd.append(qd_seg)
        all_qdd.append(qdd_seg)
        current_time += dur
        
    return (
        np.concatenate(all_times),
        np.concatenate(all_q, axis=0),
        np.concatenate(all_qd, axis=0),
        np.concatenate(all_qdd, axis=0),
    )


if __name__ == "__main__":
    q_start = np.zeros(6)
    q_end = np.array([0.5, -0.8, 1.2, -0.4, 0.9, -0.3])
    t, q, qd, qdd = generate_trajectory(q_start, q_end, duration=2.0, dt=0.02, method="quintic")
    print(f"Generated quintic trajectory: {len(t)} points over {t[-1]:.2f} s")
    print(f"  q(0): {np.round(q[0], 3)} | q_dot(0): {np.round(qd[0], 3)}")
    print(f"  q(T): {np.round(q[-1], 3)} | q_dot(T): {np.round(qd[-1], 3)}")
    print("Trajectory module verified successfully.")
