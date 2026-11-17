"""M4: pick-and-place state machine (approach, grasp, lift, move, place, retreat).

Pipeline per target pose:  IK (M3)  ->  quintic joint trajectory  ->  send to CoppeliaSim.

Usage:
    python pick_place.py                       # live run, analytical IK
    python pick_place.py --ik dls              # live run, damped least squares IK
    python pick_place.py --dry-run --ik dls    # no simulator, just checks the logic

Conventions: joint angles = NumPy (6,) in radians, poses = 4x4 in metres, base frame.

ASSUMPTIONS to confirm with teammates (rename here if their names differ):
    zmq_client.connect()              -> (client, sim)           [from the existing stub]
    zmq_client.get_joint_angles(...)  -> np.ndarray (6,)         [M1]
    zmq_client.set_joint_angles(..., q)                          [M1]
    zmq_client.open_gripper(...) / close_gripper(...)            [M1]
    ik_analytical.solve(T, q_seed)    -> np.ndarray (6,) or None [M3]
    ik_dls.solve(T, q_seed)           -> np.ndarray (6,) or None [M3]
    dh_fk.forward_kinematics(q)       -> 4x4 pose                [M2]
"""
import argparse
import time
from enum import Enum, auto

import numpy as np

from trajectory import plan_joint_trajectory

# ---------------------------------------------------------------- settings
HOME_Q = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])
APPROACH_HEIGHT = 0.15   # m above the object / drop spot
LIFT_HEIGHT = 0.20       # m above the table after grasping
GRIPPER_WAIT = 0.5       # s to let the gripper finish
DT = 0.05                # s between joint commands
TOL = 0.01               # m, success threshold for position error
# Gripper pointing straight down (tool z-axis = -z of base frame)
R_DOWN = np.array([[1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0]])


class State(Enum):
    HOME = auto()
    APPROACH = auto()
    DESCEND = auto()
    GRASP = auto()
    LIFT = auto()
    MOVE = auto()
    PLACE = auto()
    RELEASE = auto()
    RETREAT = auto()
    DONE = auto()
    FAILED = auto()


def make_pose(xyz, R=R_DOWN):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


# ---------------------------------------------------------------- robot wrappers
class SimRobot:
    """Thin wrapper over M1's zmq_client. Edit the calls here if M1's names differ."""

    def __init__(self):
        import zmq_client
        self.mod = zmq_client
        self.client, self.sim = zmq_client.connect()

    def get_q(self):
        return np.asarray(self.mod.get_joint_angles(self.sim), dtype=float)

    def set_q(self, q):
        self.mod.set_joint_angles(self.sim, q)

    def open_gripper(self):
        self.mod.open_gripper(self.sim)

    def close_gripper(self):
        self.mod.close_gripper(self.sim)

    def step(self):
        """Advance the simulation one step if stepping mode is on."""
        try:
            self.client.step()
        except Exception:
            time.sleep(DT)


class DryRunRobot:
    """Fake robot so the state machine can be tested without CoppeliaSim."""

    def __init__(self):
        self.q = HOME_Q.copy()

    def get_q(self):
        return self.q.copy()

    def set_q(self, q):
        self.q = np.asarray(q, dtype=float)

    def open_gripper(self):
        print("  [gripper] open")

    def close_gripper(self):
        print("  [gripper] close")

    def step(self):
        pass


def get_ik(method):
    """IK switch used by M5's experiments: 'analytical' or 'dls'."""
    if method == "analytical":
        from ik_analytical import solve
    elif method == "dls":
        from ik_dls import solve
    else:
        raise ValueError("ik method must be 'analytical' or 'dls'")
    return solve


# ---------------------------------------------------------------- core logic
def move_to(robot, ik, T_target, duration=None, traj_method="quintic"):
    """IK -> trajectory -> execute. Returns True if the target was reachable."""
    q_now = robot.get_q()
    q_goal = ik(T_target, q_now)
    if q_goal is None:
        return False
    traj = plan_joint_trajectory(q_now, q_goal, duration, method=traj_method)
    t_axis, Q = traj.sample(DT)
    for q in Q:
        robot.set_q(q)
        robot.step()
    return True


def run_cycle(robot, ik, obj_xyz, place_xyz, fk=None, traj_method="quintic", verbose=True):
    """Run one full pick-and-place. Returns a result dict for M5 to log.

    Keys: success, failed_state, pos_error (m), cycle_time (s)
    """
    obj_xyz = np.asarray(obj_xyz, float)
    place_xyz = np.asarray(place_xyz, float)
    up = lambda p, h: p + np.array([0.0, 0.0, h])

    # (state, target pose, gripper action after arriving)
    plan = [
        (State.APPROACH, make_pose(up(obj_xyz, APPROACH_HEIGHT)), None),
        (State.DESCEND, make_pose(obj_xyz), None),
        (State.GRASP, None, "close"),
        (State.LIFT, make_pose(up(obj_xyz, LIFT_HEIGHT)), None),
        (State.MOVE, make_pose(up(place_xyz, LIFT_HEIGHT)), None),
        (State.PLACE, make_pose(place_xyz), None),
        (State.RELEASE, None, "open"),
        (State.RETREAT, make_pose(up(place_xyz, APPROACH_HEIGHT)), None),
    ]

    result = {"success": False, "failed_state": None, "pos_error": np.nan, "cycle_time": np.nan}
    t0 = time.time()
    robot.open_gripper()

    for state, T_target, action in plan:
        if verbose:
            print(f"[{state.name}]")
        if T_target is not None and not move_to(robot, ik, T_target, traj_method=traj_method):
            result["failed_state"] = state.name
            if verbose:
                print(f"  IK failed, target unreachable -> {State.FAILED.name}")
            result["cycle_time"] = time.time() - t0
            return result
        if action == "close":
            robot.close_gripper()
            time.sleep(GRIPPER_WAIT)
        elif action == "open":
            robot.open_gripper()
            time.sleep(GRIPPER_WAIT)
        # Record position error at the grasp and place poses (needs M2's FK)
        if fk is not None and state in (State.DESCEND, State.PLACE):
            reached = fk(robot.get_q())[:3, 3]
            err = np.linalg.norm(reached - (obj_xyz if state == State.DESCEND else place_xyz))
            result["pos_error"] = err if np.isnan(result["pos_error"]) else max(result["pos_error"], err)

    result["cycle_time"] = time.time() - t0
    result["success"] = bool(np.isnan(result["pos_error"]) or result["pos_error"] < TOL)
    if verbose:
        print(f"[{State.DONE.name}] success={result['success']}  "
              f"error={result['pos_error']:.4f} m  time={result['cycle_time']:.2f} s")
    return result


# ---------------------------------------------------------------- entry point
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ik", choices=["analytical", "dls"], default="analytical")
    ap.add_argument("--traj", choices=["cubic", "quintic"], default="quintic")
    ap.add_argument("--dry-run", action="store_true", help="no simulator")
    ap.add_argument("--obj", type=float, nargs=3, default=[0.4, 0.2, 0.05])
    ap.add_argument("--place", type=float, nargs=3, default=[0.4, -0.2, 0.05])
    args = ap.parse_args()

    robot = DryRunRobot() if args.dry_run else SimRobot()
    ik = get_ik(args.ik)
    try:
        from dh_fk import forward_kinematics as fk
    except ImportError:
        fk = None
    run_cycle(robot, ik, args.obj, args.place, fk=fk, traj_method=args.traj)