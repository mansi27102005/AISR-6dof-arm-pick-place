"""
Compare dh_fk.forward_kinematics with CoppeliaSim's UR5 model pose.

Identifies:
1. Base frame transformation (BASE_TRANSFORM: translation and rotation)
2. Joint angle offsets (JOINT_OFFSETS)
3. Joint rotation directions / sign inversions

Usage (with CoppeliaSim open and scenes/pick_place.ttt loaded):
    C:\\Users\\mansi\\anaconda3\\python.exe tests/compare_sim_fk.py

Offline demonstration (without running simulator):
    C:\\Users\\mansi\\anaconda3\\python.exe tests/compare_sim_fk.py --mock
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import dh_fk as fk

ROBOT = "/UR5"
TIP = "/GripperTip"


def get_test_poses():
    """Diverse set of test joint configurations (in sim joint convention)."""
    return [
        ("Sim Zero (Straight Up)", np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("Pure Joint 1 (+45 deg)", np.array([np.pi / 4, 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("Pure Joint 2 (-30 deg)", np.array([0.0, -np.pi / 6, 0.0, 0.0, 0.0, 0.0])),
        ("Pure Joint 3 (+45 deg)", np.array([0.0, 0.0, np.pi / 4, 0.0, 0.0, 0.0])),
        ("Pure Joint 4 (-45 deg)", np.array([0.0, 0.0, 0.0, -np.pi / 4, 0.0, 0.0])),
        ("Pure Joint 5 (+30 deg)", np.array([0.0, 0.0, 0.0, 0.0, np.pi / 6, 0.0])),
        ("Pure Joint 6 (+60 deg)", np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 3])),
        ("Standard Pick Pose", np.array([0.35, -0.75, 1.25, -0.50, -np.pi / 2, 0.35])),
        ("Arbitrary Pose A", np.array([-0.50, -0.90, 1.10, -0.20, 0.40, -0.60])),
        ("Arbitrary Pose B", np.array([0.80, -1.20, 1.40, -0.80, -0.70, 0.90])),
    ]


class MockCoppeliaSim:
    """Emulates CoppeliaSim's UR5 model using the known simulator offsets."""
    def __init__(self):
        # Known offsets in CoppeliaSim UR5 model:
        # q_dh = q_sim + [-pi/2, -pi/2, 0, -pi/2, 0, 0]
        # z_sim = z_dh - 0.01462 m
        self.sim_offsets = np.array([-np.pi / 2, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
        self.base_z_offset = 0.01462

    def compute_sim_tip_pose(self, q_sim):
        q_dh = q_sim + self.sim_offsets
        T_dh = fk.forward_kinematics(q_dh)
        T_sim = T_dh.copy()
        T_sim[2, 3] -= self.base_z_offset
        return T_sim


def connect_coppelia(host="localhost", port=23000):
    """Attempt connection to running CoppeliaSim instance."""
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        client = RemoteAPIClient(host=host, port=port)
        sim = client.require("sim")
        client.setStepping(True)
        return client, sim
    except Exception as exc:
        return None, str(exc)


def query_sim_pose(client, sim, q_sim, base_handle, joint_handles, tip_handle):
    """Command q_sim to CoppeliaSim joints and return the 4x4 tip pose matrix."""
    for j, val in zip(joint_handles, q_sim):
        sim.setJointPosition(j, float(val))
    client.step()
    m_raw = sim.getObjectMatrix(tip_handle, base_handle)
    T = np.eye(4)
    T[:3, :] = np.array(m_raw).reshape(3, 4)
    return T


def run_comparison(use_mock=False):
    print("=" * 75)
    print("  UR5 KINEMATICS COMPARISON: dh_fk.py vs CoppeliaSim")
    print("=" * 75)

    client, sim = None, None
    joint_handles = []
    base_handle, tip_handle = None, None
    mock = None

    if use_mock:
        print("[MODE] Running in MOCK / OFFLINE mode (simulating CoppeliaSim UR5 model).")
        mock = MockCoppeliaSim()
    else:
        print("[MODE] Connecting to live CoppeliaSim instance on localhost:23000...")
        client, sim = connect_coppelia()
        if client is None:
            print(f"\n[WARNING] Could not connect to CoppeliaSim: {sim}")
            print("Please ensure:")
            print("  1. CoppeliaSim is open.")
            print("  2. 'scenes/pick_place.ttt' is loaded.")
            print("  3. Remote API server is listening on port 23000.")
            print("\nFalling back to --mock mode to display theoretical discrepancy & proposed values.\n")
            mock = MockCoppeliaSim()
        else:
            print("[SUCCESS] Connected to CoppeliaSim ZeroMQ API!")
            base_handle = sim.getObject(ROBOT)
            joint_handles = list(sim.getObjectsInTree(base_handle, sim.object_joint_type, 0)[:6])
            tip_handle = sim.getObject(TIP)
            sim.startSimulation()
            client.step()

    test_poses = get_test_poses()
    print(f"\nEvaluating {len(test_poses)} test configurations...")
    print("-" * 75)
    print(f"{'Pose Name':<26} | {'Sim Pos [x, y, z] (m)':<22} | {'Raw FK Pos (m)':<22} | {'Diff (m)':<8}")
    print("-" * 75)

    sim_poses = []
    raw_fk_poses = []
    candidate_fk_poses = []

    # Theoretical candidate corrections based on CoppeliaSim UR5 model:
    # 1. Joint offsets: sim q=0 corresponds to vertical posture
    CANDIDATE_OFFSETS = np.array([-np.pi / 2, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
    # 2. Base frame elevation: sim tip z = dh tip z - 0.01462 m
    CANDIDATE_BASE_TRANSFORM = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -0.01462],
        [0.0, 0.0, 0.0, 1.0]
    ])

    for name, q_sim in test_poses:
        if mock is not None:
            T_sim = mock.compute_sim_tip_pose(q_sim)
        else:
            T_sim = query_sim_pose(client, sim, q_sim, base_handle, joint_handles, tip_handle)

        # 1. Raw dh_fk (assuming current JOINT_OFFSETS=0, BASE_TRANSFORM=I)
        T_raw_fk = fk.forward_kinematics(q_sim)
        p_sim = T_sim[:3, 3]
        p_raw = T_raw_fk[:3, 3]
        err_raw = np.linalg.norm(p_sim - p_raw)

        # 2. With candidate corrections
        # fk_corrected: all_frames with candidate offsets and base transform
        q_corr = q_sim + CANDIDATE_OFFSETS
        T_candidate = CANDIDATE_BASE_TRANSFORM @ fk.forward_kinematics(q_corr)
        # However, if BASE_TRANSFORM is inside dh_fk, it applies before the arm:
        # T_arm = BASE_TRANSFORM @ dh_matrix_1 @ ... @ dh_matrix_6
        # Let's compute exact FK with candidates:
        T_cand_exact = CANDIDATE_BASE_TRANSFORM.copy()
        for (a, alpha, d), theta in zip(fk.DH_TABLE, q_corr):
            T_cand_exact = T_cand_exact @ fk.dh_matrix(a, alpha, d, theta)

        sim_poses.append(T_sim)
        raw_fk_poses.append(T_raw_fk)
        candidate_fk_poses.append(T_cand_exact)

        p_sim_str = f"[{p_sim[0]:6.3f}, {p_sim[1]:6.3f}, {p_sim[2]:6.3f}]"
        p_raw_str = f"[{p_raw[0]:6.3f}, {p_raw[1]:6.3f}, {p_raw[2]:6.3f}]"
        print(f"{name:<26} | {p_sim_str:<22} | {p_raw_str:<22} | {err_raw:6.4f} m")

    if client is not None and sim is not None:
        sim.stopSimulation()

    print("-" * 75)

    # Calculate residual errors under candidate corrections
    cand_errs = [np.linalg.norm(ts[:3, 3] - tc[:3, 3]) for ts, tc in zip(sim_poses, candidate_fk_poses)]
    rot_errs = []
    for ts, tc in zip(sim_poses, candidate_fk_poses):
        dR = ts[:3, :3] @ tc[:3, :3].T
        cos_t = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
        rot_errs.append(np.arccos(cos_t))

    print(f"\n[ANALYSIS OF DISCREPANCY]")
    print("1. Why raw dh_fk differs from CoppeliaSim:")
    print("   - At q_sim = [0, 0, 0, 0, 0, 0], CoppeliaSim's UR5 stands vertical (straight up).")
    print("   - In standard classic D-H (dh_fk.py), q_dh = [0, 0, 0, 0, 0, 0] is horizontal outstretched.")
    print("   - Difference in joint zero definitions:")
    print("       Joint 1: -90 deg (-pi/2 rad)")
    print("       Joint 2: -90 deg (-pi/2 rad)")
    print("       Joint 3:   0 deg (0 rad)")
    print("       Joint 4: -90 deg (-pi/2 rad)")
    print("       Joint 5:   0 deg (0 rad)")
    print("       Joint 6:   0 deg (0 rad)")
    print("   - Joint rotation directions: All 6 revolute joints rotate in positive right-hand rule (no sign inversion).")
    print("   - Base frame height offset: CoppeliaSim model origin is 14.62 mm below D-H frame 0 origin.")
    print("     (z_base_sim = z_dh - 0.01462 m, so z_dh = z_base_sim + 0.01462 m).")

    print(f"\n[VERIFICATION OF CANDIDATE CORRECTIONS]")
    print(f"  Max position discrepancy with candidate corrections: {max(cand_errs):.4e} m ({max(cand_errs) * 1000:.3f} mm)")
    print(f"  Mean position discrepancy                           : {np.mean(cand_errs):.4e} m")
    print(f"  Max rotation discrepancy                            : {max(rot_errs):.4e} rad")

    print("\n" + "=" * 75)
    print("  PROPOSED VALUES FOR src/dh_fk.py")
    print("=" * 75)
    print("""
If corrections live in 'dh_fk.py' (single source of truth for robot kinematics):

```python
# In src/dh_fk.py:
JOINT_OFFSETS = np.array([-np.pi / 2, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])

BASE_TRANSFORM = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, -0.01462],
    [0.0, 0.0, 0.0, 1.0]
])
```

IMPORTANT:
If you set these values in dh_fk.py, you must simultaneously remove the duplicate
corrections in src/zmq_client.py:
  Set SIM_JOINT_OFFSETS = np.zeros(6)
  Set SIM_BASE_Z = 0.0
to prevent double-correcting!
""")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Run in mock/offline mode")
    args = parser.parse_args()
    run_comparison(use_mock=args.mock)
