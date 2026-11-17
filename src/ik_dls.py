"""
M3: numerical inverse kinematics with damped least squares (DLS).

Each iteration solves   dq = J^T (J J^T + lambda^2 I)^{-1} e
where e stacks the position error and the axis-angle rotation error of
R_target R(q)^T, both expressed in the base frame (matching dh_fk.jacobian).

lambda stays at lam0 while the arm is well conditioned and grows towards
lam_max as dh_fk.manipulability() falls below w_ref, so steps stay bounded
near singularities instead of being amplified by the vanishing Jacobian.
As the pose error shrinks the damping is scaled down as well (never below
1e-4 of its value), because a constant lambda biases the fixed point and
would stall tight convergence near singular targets.  The step length is
capped at max_step radians per iteration.

A single damped least squares run can stall in a local minimum when it is
started far from the target, so after the first start the solver retries
from the fixed MULTI_START_SEEDS postures (hard-coded for reproducibility)
and keeps the best result.  inverse_kinematics(T) returns the 6 joint
angles; RuntimeError is raised only if no start converges (or the best
effort is returned with raise_on_fail=False).  return_info=True also
returns a dict with iterations, final errors, damping and manipulability.
"""
import numpy as np

import dh_fk as fk

MULTI_START_SEEDS = (
    np.array([-2.3338, -0.0045, 0.6377, -2.9613, -2.2121, 2.6905]),
    np.array([-0.8398, -1.8894, -2.5852, 0.9625, -0.2555, 3.0642]),
    np.array([-0.3541, 0.4303, 2.5642, -1.5441, 0.5578, -0.8852]),
    np.array([-2.37149084, -1.51982037, -0.59205998, 2.94796969,
              -2.12172408, 2.24494236]),
    np.array([-1.84571515, 1.27666506, 1.18444012, 1.18669938,
              1.02063114, -2.37342447]),
    np.array([2.78679692, -0.8832837, 1.78948518, 0.57351775,
              -1.29227176, 2.65606382]),
    np.array([-1.79427403, -0.52262391, 1.93330621, -1.42048195,
              1.98410131, -2.46531417]),
    np.array([2.25782716, 0.98506923, -0.11444864, 1.01343167,
              0.20755203, 0.9293177]),
    np.array([-1.34809162, 1.60390018, -1.6176439, -1.30295436,
              2.1812401, -1.56038518]),
    np.array([-0.38703801, 2.49464878, 1.72706762, 0.84662056,
              2.11156666, -1.2185353]),
)


def _wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def _check_pose(T):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"expected a 4x4 pose, got shape {T.shape}")
    if not np.all(np.isfinite(T)):
        raise ValueError("pose contains non-finite values")
    return T


def _check_q(q, name="q"):
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != fk.N_JOINTS:
        raise ValueError(f"expected {fk.N_JOINTS} joint angles, got {q.size}")
    return q


def damping(lam0, lam_max, w, w_ref):
    """Singularity-aware damping: lam0 for w >= w_ref, rising to lam_max as w -> 0."""
    w = float(w)
    if w >= w_ref:
        return float(lam0)
    gap = 1.0 - w / w_ref
    return float(lam0 + (lam_max - lam0) * gap * gap)


def _axis_angle(dR):
    """Rotation vector (axis * angle) of a rotation matrix."""
    cos_t = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    t = np.arccos(cos_t)
    if t < 1e-12:
        return np.zeros(3)
    s = np.sin(t)
    if s > 1e-8:
        v = np.array([dR[2, 1] - dR[1, 2],
                      dR[0, 2] - dR[2, 0],
                      dR[1, 0] - dR[0, 1]]) / (2.0 * s)
        return t * v
    M = (dR + np.eye(3)) / 2.0
    i = int(np.argmax(np.diag(M)))
    axis = np.zeros(3)
    axis[i] = np.sqrt(max(M[i, i], 0.0))
    if axis[i] > 1e-12:
        for j in range(3):
            if j != i:
                axis[j] = M[j, i] / axis[i]
    n = np.linalg.norm(axis)
    if n > 1e-12:
        axis /= n
    return t * axis


def pose_error(T_target, q):
    """Position and rotation error vectors of FK(q) against T_target (base frame)."""
    T = fk.forward_kinematics(q)
    dp = T_target[:3, 3] - T[:3, 3]
    dw = _axis_angle(T_target[:3, :3] @ T[:3, :3].T)
    return dp, dw


def _solve_one(T, start, max_iter, pos_tol, rot_tol, lam0, lam_max, w_ref, max_step):
    q = np.asarray(start, dtype=float).copy()
    w = fk.manipulability(q)
    lam = damping(lam0, lam_max, w, w_ref)
    lam_eff = lam
    dp, dw = pose_error(T, q)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        dp, dw = pose_error(T, q)
        if np.linalg.norm(dp) <= pos_tol and np.linalg.norm(dw) <= rot_tol:
            converged = True
            break
        J = fk.jacobian(q)
        w = fk.manipulability(q)
        lam = damping(lam0, lam_max, w, w_ref)
        e = np.concatenate([dp, dw])
        # shrink the damping as the error shrinks: a constant lambda biases the
        # fixed point and stalls tight convergence near singular targets
        scale = float(np.clip(np.linalg.norm(e), 1e-4, 1.0))
        lam_eff = lam * scale
        A = J @ J.T + (lam_eff * lam_eff) * np.eye(fk.N_JOINTS)
        dq = J.T @ np.linalg.solve(A, e)
        step = float(np.max(np.abs(dq)))
        if step > max_step:
            dq *= max_step / step
        q = _wrap(q + dq)
    if not converged:
        dp, dw = pose_error(T, q)
    info = {
        "iterations": it,
        "converged": converged,
        "position_error": float(np.linalg.norm(dp)),
        "rotation_error": float(np.linalg.norm(dw)),
        "lambda": lam,
        "lambda_effective": lam_eff,
        "manipulability": w,
    }
    return q, info


def inverse_kinematics(T, q0=None, max_iter=300, pos_tol=1e-8, rot_tol=1e-8,
                       lam0=1e-3, lam_max=0.5, w_ref=1e-4, max_step=0.5,
                       multi_start=True, return_info=False, raise_on_fail=True):
    """Numerical IK: pose in, 6 joint angles out (damped least squares)."""
    T = _check_pose(T)
    first = np.zeros(fk.N_JOINTS) if q0 is None else _check_q(q0)
    starts = [first]
    if multi_start:
        for seed in MULTI_START_SEEDS:
            if not np.allclose(seed, first, atol=1e-12):
                starts.append(seed)
    best_q, best_info, best_score = None, None, None
    tried = 0
    for k, start in enumerate(starts):
        tried = k + 1
        q, info = _solve_one(T, start, max_iter, pos_tol, rot_tol,
                             lam0, lam_max, w_ref, max_step)
        info["starts_tried"] = tried
        score = (info["converged"],
                 -(info["position_error"] + info["rotation_error"]))
        if best_score is None or score > best_score:
            best_q, best_info, best_score = q, info, score
        if info["converged"]:
            break
    if not best_info["converged"] and raise_on_fail:
        raise RuntimeError(
            f"DLS IK did not converge from any of {tried} start(s): "
            f"best position error {best_info['position_error']:.3e} m, "
            f"rotation error {best_info['rotation_error']:.3e} rad"
        )
    if return_info:
        return best_q, best_info
    return best_q


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    q_true = np.array([0.3, -1.0, 1.2, 0.5, -0.8, 1.1])
    T = fk.forward_kinematics(q_true)
    q, info = inverse_kinematics(T, return_info=True)
    print("IK solution:   ", q)
    print("true joints:   ", q_true)
    print("iterations:    ", info["iterations"], "starts:", info["starts_tried"])
    print("position error:", info["position_error"], "m")
    print("rotation error:", info["rotation_error"], "rad")
