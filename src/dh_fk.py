"""
M2: D-H table, forward kinematics and Jacobian for the UR5.

Convention: standard (classic) Denavit-Hartenberg.
Each link transform is  T_i = Rz(theta_i) * Tz(d_i) * Tx(a_i) * Rx(alpha_i).

Units: metres and radians. Poses are 4x4 homogeneous transforms in the robot base frame.
Joint angles are NumPy arrays of 6 values.

UR5 D-H parameters are from Universal Robots' published values for the UR5
(not the UR5e, which has different numbers).
"""
import numpy as np

# One row per joint: (a [m], alpha [rad], d [m]). theta is the joint variable.
DH_TABLE = np.array([
    [ 0.0,      np.pi / 2, 0.089159],   # joint 1 (shoulder pan)
    [-0.425,    0.0,       0.0     ],   # joint 2 (shoulder lift)
    [-0.39225,  0.0,       0.0     ],   # joint 3 (elbow)
    [ 0.0,      np.pi / 2, 0.10915 ],   # joint 4 (wrist 1)
    [ 0.0,     -np.pi / 2, 0.09465 ],   # joint 5 (wrist 2)
    [ 0.0,      0.0,       0.0823  ],   # joint 6 (wrist 3)
])

N_JOINTS = 6

# Optional corrections to match CoppeliaSim's UR5 model (set after comparing with the sim).
# joint_offsets are added to every joint angle, base_transform is applied before the arm.
JOINT_OFFSETS = np.zeros(N_JOINTS)
BASE_TRANSFORM = np.eye(4)


def dh_matrix(a, alpha, d, theta):
    """4x4 transform from frame i-1 to frame i for one set of D-H parameters."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def _check_q(q):
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.size != N_JOINTS:
        raise ValueError(f"expected {N_JOINTS} joint angles, got {q.size}")
    return q


def all_frames(q):
    """Return a list of 7 poses: base frame, then frame after joint 1, ..., joint 6."""
    q = _check_q(q) + JOINT_OFFSETS
    T = BASE_TRANSFORM.copy()
    frames = [T.copy()]
    for (a, alpha, d), theta in zip(DH_TABLE, q):
        T = T @ dh_matrix(a, alpha, d, theta)
        frames.append(T.copy())
    return frames


def forward_kinematics(q):
    """4x4 pose of the end effector (flange) for joint angles q (radians)."""
    return all_frames(q)[-1]


def position_and_rotation(q):
    """Convenience: return (position as 3-vector, rotation as 3x3)."""
    T = forward_kinematics(q)
    return T[:3, 3].copy(), T[:3, :3].copy()


def jacobian(q):
    """
    6x6 geometric Jacobian in the base frame.
    Top 3 rows: linear velocity of the end effector. Bottom 3 rows: angular velocity.
    Column i = [ z_{i-1} x (p_e - p_{i-1}) ; z_{i-1} ] for a revolute joint.
    """
    frames = all_frames(q)
    p_e = frames[-1][:3, 3]
    J = np.zeros((6, N_JOINTS))
    for i in range(N_JOINTS):
        z = frames[i][:3, 2]
        p = frames[i][:3, 3]
        J[:3, i] = np.cross(z, p_e - p)
        J[3:, i] = z
    return J


def manipulability(q):
    """Yoshikawa measure sqrt(det(J J^T)). Near zero means close to a singularity."""
    J = jacobian(q)
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    q0 = np.zeros(N_JOINTS)
    print("Pose at q = 0:\n", forward_kinematics(q0))
    print("Manipulability at q = 0:", manipulability(q0))
