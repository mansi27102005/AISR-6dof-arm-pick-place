"""
M3: closed-form inverse kinematics for the UR5 (standard D-H, as in dh_fk.py).

Geometric solution producing up to 8 configurations per reachable pose:

1. Wrist centre p_w = p_e - d6 * z_e.  Expressed in the frame of joint 1 its
   lateral coordinate equals the shoulder offset d4, which gives
   sin(q1 - psi) = d4 / hypot(p_wx, p_wy)  ->  2 shoulder solutions.
2. R = R01(q1)^T @ R06 = Rz(q2+q3+q4) @ Ry(-q5) @ Rz(q6) yields
   q5 = +/- acos(R[2, 2]) and, when sin(q5) != 0, q6 and q2+q3+q4;
   at the wrist singularity q6 is fixed to 0 and q2+q3+q4 taken directly.
3. Removing the known d5 * (sin(q234), -cos(q234)) offset from the wrist
   centre leaves a planar two-link problem for q2 and q3 (elbow up/down),
   then q4 = q234 - q2 - q3.

inverse_kinematics(T) returns the 6 joint angles closest to q_prefer
(default zeros).  all_solutions(T) returns every configuration found.
ValueError is raised when no configuration reaches the target pose.
"""
import numpy as np

import dh_fk as fk

_DH = fk.DH_TABLE
_A2, _A3 = _DH[1, 0], _DH[2, 0]
_D1, _D4, _D5, _D6 = _DH[0, 2], _DH[3, 2], _DH[4, 2], _DH[5, 2]
_SINGULAR_EPS = 1e-9
_REACH_TOL = 1e-9


def _wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def _Ry(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _check_pose(T):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"expected a 4x4 pose, got shape {T.shape}")
    if not np.all(np.isfinite(T)):
        raise ValueError("pose contains non-finite values")
    return T


def all_solutions(T):
    """Every configuration that reaches pose T, as a list of 6-angle arrays."""
    T = _check_pose(T)
    T_arm = np.linalg.inv(fk.BASE_TRANSFORM) @ T
    R_e, p_e = T_arm[:3, :3], T_arm[:3, 3]
    p_w = p_e - _D6 * R_e[:, 2]
    r = float(np.hypot(p_w[0], p_w[1]))
    if r < abs(_D4) - _REACH_TOL:
        raise ValueError(
            f"target out of reach: wrist-centre radius {r:.4f} m "
            f"is smaller than shoulder offset d4 = {abs(_D4):.4f} m"
        )
    psi = np.arctan2(p_w[1], p_w[0])
    swing = np.arcsin(np.clip(_D4 / max(r, _REACH_TOL), -1.0, 1.0))
    solutions = []
    failed_branches = 0
    for q1 in (psi + np.pi - swing, psi + swing):
        c1, s1 = np.cos(q1), np.sin(q1)
        R01 = fk.dh_matrix(*_DH[0], q1)
        Rbar = R01[:3, :3].T @ R_e
        q5_pos = np.arccos(np.clip(Rbar[2, 2], -1.0, 1.0))
        for q5 in (q5_pos, -q5_pos):
            s5 = np.sin(q5)
            if abs(s5) > _SINGULAR_EPS:
                q234 = np.arctan2(-Rbar[1, 2] / s5, -Rbar[0, 2] / s5)
                q6 = np.arctan2(-Rbar[2, 1] / s5, Rbar[2, 0] / s5)
            else:
                q6 = 0.0
                Rz = Rbar @ _Ry(q5)
                q234 = np.arctan2(Rz[1, 0], Rz[0, 0])
            x1 = c1 * p_w[0] + s1 * p_w[1]
            y1 = p_w[2] - _D1
            X = x1 - _D5 * np.sin(q234)
            Y = y1 + _D5 * np.cos(q234)
            c3 = (X * X + Y * Y - _A2 ** 2 - _A3 ** 2) / (2 * _A2 * _A3)
            if abs(c3) > 1.0 + _REACH_TOL:
                failed_branches += 1
                continue
            c3 = np.clip(c3, -1.0, 1.0)
            for sign in (1.0, -1.0):
                q3 = sign * np.arccos(c3)
                q2 = np.arctan2(Y, X) - np.arctan2(_A3 * np.sin(q3),
                                                   _A2 + _A3 * np.cos(q3))
                q4 = _wrap(q234 - q2 - q3)
                q = _wrap(np.array([q1, q2, q3, q4, q5, q6]) - fk.JOINT_OFFSETS)
                solutions.append(q)
    if not solutions:
        raise ValueError(
            f"no configuration reaches the target pose "
            f"({failed_branches} of 4 shoulder/wrist branches out of reach)"
        )
    return solutions


def inverse_kinematics(T, q_prefer=None):
    """Closed-form IK: pose in, 6 joint angles out (closest to q_prefer)."""
    solutions = all_solutions(T)
    if q_prefer is None:
        prefer = np.zeros(fk.N_JOINTS)
    else:
        prefer = np.asarray(q_prefer, dtype=float).reshape(-1)
        if prefer.size != fk.N_JOINTS:
            raise ValueError(f"q_prefer must have {fk.N_JOINTS} values, "
                             f"got {prefer.size}")
    costs = [float(np.sum(_wrap(s - prefer) ** 2)) for s in solutions]
    return solutions[int(np.argmin(costs))].copy()


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    q_true = np.array([0.3, -1.0, 1.2, 0.5, -0.8, 1.1])
    T = fk.forward_kinematics(q_true)
    q = inverse_kinematics(T, q_prefer=q_true)
    print("IK solution:   ", q)
    print("true joints:   ", q_true)
    print("max pose error:", np.abs(fk.forward_kinematics(q) - T).max())
