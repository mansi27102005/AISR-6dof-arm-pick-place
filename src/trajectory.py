"""M4: joint-space trajectory generation (cubic / quintic).

Interface (agreed team conventions):
    joint angles : NumPy array of 6 values, radians
    time         : seconds

Main entry point:
    traj = plan_joint_trajectory(q_start, q_end, duration, method="quintic")
    q, qd, qdd = traj.at(t)          # evaluate at time t
    t, Q = traj.sample(dt=0.02)      # sample the whole path, Q has shape (N, 6)
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


if __name__ == "__main__":
    # Quick self-check with plain numbers, no simulator needed
    q0 = np.zeros(6)
    q1 = np.array([0.5, -0.8, 1.0, -0.3, 0.2, 1.2])
    for m in ("cubic", "quintic"):
        tr = plan_joint_trajectory(q0, q1, duration=2.0, method=m)
        q_a, v_a, a_a = tr.at(0.0)
        q_b, v_b, a_b = tr.at(2.0)
        print(f"{m}: start ok={np.allclose(q_a, q0)}, end ok={np.allclose(q_b, q1)}, "
              f"v(0)={np.abs(v_a).max():.1e}, v(T)={np.abs(v_b).max():.1e}, "
              f"a(0)={np.abs(a_a).max():.1e}, a(T)={np.abs(a_b).max():.1e}")
