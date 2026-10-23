"""M2: D-H table and forward kinematics for the UR5."""
import numpy as np

# UR5 standard D-H parameters: (a, alpha, d). Verify against your source before using.
DH = [
    (0.0,      np.pi / 2,  0.089159),
    (-0.425,   0.0,        0.0),
    (-0.39225, 0.0,        0.0),
    (0.0,      np.pi / 2,  0.10915),
    (0.0,     -np.pi / 2,  0.09465),
    (0.0,      0.0,        0.0823),
]


def dh_matrix(a, alpha, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,       ca,      d],
        [0,   0,        0,       1],
    ])


def forward_kinematics(q):
    T = np.eye(4)
    for (a, alpha, d), theta in zip(DH, q):
        T = T @ dh_matrix(a, alpha, d, theta)
    return T
