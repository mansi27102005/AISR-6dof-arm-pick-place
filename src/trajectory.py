"""M4: joint-space trajectory generation (cubic / quintic).

Interface (agreed team conventions):
    joint angles : NumPy array of 6 values, radians
    time         : seconds

Class-based entry point (M4):
    traj = plan_joint_trajectory(q_start, q_end, duration, method="quintic")
    q, qd, qdd = traj.at(t)          # evaluate at time t
    t, Q = traj.sample(dt=0.02)      # sample the whole path, Q has shape (N, 6)

Procedural entry point (M5):
    times, q, qd, qdd = generate_trajectory(q_start, q_end, duration=2.0, dt=0.02, method="quintic")
    times, q, qd, qdd = plan_multisegment_trajectory(waypoints, durations, dt=0.02, method="quintic")
"""
import numpy as np


def _poly_coeffs(method):
    """Return (position, velocity, acceleration) coefficient builders for s(tau), tau in [0, 1]."""
    if method == "cubic":
        # s = 3t^2 - 2t^3  -> zero velocity at both ends
        return (np.array([0, 0, 3, -2]), 3)
    if method == "quintic":
        # s = 10t^3 - 15t^4 + 6t^5 -> zero velocity AND acceleration at both ends
        return (np.array([0, 0, 0, 10, -15, 6]), 5)
    raise ValueError(f"method must be 'cubic' or 'quintic', got {method!r}")


class JointTrajectory:
    """Point-to-point trajectory in joint space with rest-to-rest boundary conditions."""

    def __init__(self, q_start, q_end, duration, method="quintic"):
        self.q0 = np.asarray(q_start, dtype=float).reshape(-1)
        self.q1 = np.asarray(q_end, dtype=float).reshape(-1)
        if self.q0.shape != self.q1.shape:
            raise ValueError("q_start and q_end must have the same length")
        if duration <= 0:
            raise ValueError("duration must be positive")
        self.T = float(duration)
        self.method = method
        self.coeffs, _ = _poly_coeffs(method)
        self.dq = self.q1 - self.q0

    def _s(self, tau):
        c = self.coeffs
        k = np.arange(len(c))
        s = np.sum(c * tau ** k)
        ds = np.sum(c[1:] * k[1:] * tau ** (k[1:] - 1))
        dds = np.sum(c[2:] * k[2:] * (k[2:] - 1) * tau ** (k[2:] - 2))
        return s, ds, dds

    def at(self, t):
        """Return (q, qd, qdd) at time t. t is clamped to [0, duration]."""
        tau = np.clip(t / self.T, 0.0, 1.0)
        s, ds, dds = self._s(tau)
        q = self.q0 + self.dq * s
        qd = self.dq * ds / self.T
        qdd = self.dq * dds / self.T ** 2
        return q, qd, qdd

    def sample(self, dt=0.02):
        """Sample the full trajectory. Returns (t, Q) with Q of shape (N, n_joints)."""
        n = max(int(np.ceil(self.T / dt)), 1)
        t = np.linspace(0.0, self.T, n + 1)
        Q = np.array([self.at(ti)[0] for ti in t])
        return t, Q


def plan_joint_trajectory(q_start, q_end, duration=None, method="quintic",
                          v_max=1.0, a_max=2.0):
    """Build a JointTrajectory. If duration is None, pick one from joint speed/accel limits.

    v_max, a_max are per-joint limits in rad/s and rad/s^2 (used only if duration is None).
    """
    dq = np.abs(np.asarray(q_end, float) - np.asarray(q_start, float))
    if duration is None:
        # Peak velocity / acceleration of the normalised profile
        if method == "quintic":
            k_v, k_a = 15.0 / 8.0, 10.0 / np.sqrt(3.0)
        else:
            k_v, k_a = 1.5, 6.0
        d = dq.max()
        duration = max(k_v * d / v_max, np.sqrt(k_a * d / a_max), 0.5)
    return JointTrajectory(q_start, q_end, duration, method)


def plan_via_points(waypoints, durations, method="quintic"):
    """Chain several rest-to-rest segments. Returns a list of JointTrajectory objects."""
    return [JointTrajectory(waypoints[i], waypoints[i + 1], durations[i], method)
            for i in range(len(waypoints) - 1)]


def cubic_step(t, t_start, t_end, q_start, q_end):
    """Evaluate cubic polynomial trajectory at time t."""
    T = t_end - t_start
    if T <= 0:
        raise ValueError("Duration must be strictly positive")
    tau = np.clip((t - t_start) / T, 0.0, 1.0)
    s = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
    ds = (6.0 * tau - 6.0 * (tau ** 2)) / T
    dds = (6.0 - 12.0 * tau) / (T ** 2)
    q0, q1 = np.asarray(q_start, float), np.asarray(q_end, float)
    dq = q1 - q0
    return q0 + dq * s, dq * ds, dq * dds


def quintic_step(t, t_start, t_end, q_start, q_end):
    """Evaluate quintic (5th order) polynomial trajectory at time t."""
    T = t_end - t_start
    if T <= 0:
        raise ValueError("Duration must be strictly positive")
    tau = np.clip((t - t_start) / T, 0.0, 1.0)
    s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)
    ds = (30.0 * (tau ** 2) - 60.0 * (tau ** 3) + 30.0 * (tau ** 4)) / T
    dds = (60.0 * tau - 180.0 * (tau ** 2) + 120.0 * (tau ** 3)) / (T ** 2)
    q0, q1 = np.asarray(q_start, float), np.asarray(q_end, float)
    dq = q1 - q0
    return q0 + dq * s, dq * ds, dq * dds


def generate_trajectory(q_start, q_end, duration=2.0, dt=0.02, method="quintic"):
    """Generate time-series arrays for a single segment motion."""
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
    """Generate smooth continuous trajectory connecting an ordered sequence of waypoints."""
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
    q0 = np.zeros(6)
    q1 = np.array([0.5, -0.8, 1.0, -0.3, 0.2, 1.2])
    for m in ("cubic", "quintic"):
        tr = plan_joint_trajectory(q0, q1, duration=2.0, method=m)
        q_a, v_a, a_a = tr.at(0.0)
        q_b, v_b, a_b = tr.at(2.0)
        print(f"{m}: start ok={np.allclose(q_a, q0)}, end ok={np.allclose(q_b, q1)}")
