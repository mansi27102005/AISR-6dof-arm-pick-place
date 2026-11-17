"""M1: connection to CoppeliaSim (ZeroMQ remote API) and robot I/O.

Functions used by pick_place.py (all take `sim` as first argument):
    connect()                  -> (client, sim)
    get_joint_angles(sim)      -> np.ndarray (6,) radians
    set_joint_angles(sim, q)
    open_gripper(sim) / close_gripper(sim)
Extra helpers:
    start_simulation(client, sim), stop_simulation(client, sim)
    get_tip_pose(sim)          -> 4x4 pose of the tool in the robot base frame
    get_object_position(sim, name) -> (3,) position in the robot base frame
    part_in_bin(sim)           success check for the experiments
    reset_scene_objects(sim)   puts the part back on the conveyor
    verify_fk(sim, n=5)        compares dh_fk against the simulator pose

Scene conventions (created by scenes/build_scene.py):
    /UR5            robot model, base at world origin
    /Part           cube to pick
    /Bin            drop bin
    /GripperTip     dummy at the tool flange
The gripper is a "magnetic" gripper: close_gripper() parents the part to
/GripperTip if it is within GRIP_RANGE, open_gripper() releases it. This keeps
the demo deterministic.
"""
import numpy as np

ROBOT = "/UR5"
TIP = "/GripperTip"
PART = "/Part"
GRIP_RANGE = 0.08      # m, max tip-to-part distance for a successful grasp
PART_START = (0.40, 0.20, 0.0425)

_cache = {}


def connect(host="localhost", port=23000, stepping=True):
    """Connect to CoppeliaSim. Start CoppeliaSim first (the ZMQ remote API is on by default)."""
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    except ImportError as e:
        raise ImportError("pip install coppeliasim-zmqremoteapi-client") from e
    client = RemoteAPIClient(host=host, port=port)
    sim = client.require("sim")
    if stepping:
        client.setStepping(True)   # one client.step() = one simulation step
    _cache.clear()
    return client, sim


def start_simulation(client, sim):
    sim.startSimulation()
    client.step()


def stop_simulation(client, sim):
    sim.stopSimulation()


# ---------------------------------------------------------------- handles
def _handles(sim):
    if "joints" not in _cache:
        base = sim.getObject(ROBOT)
        joints = sim.getObjectsInTree(base, sim.object_joint_type, 0)
        if len(joints) < 6:
            raise RuntimeError(f"{ROBOT} has {len(joints)} joints, expected 6")
        _cache["base"] = base
        _cache["joints"] = list(joints[:6])   # depth-first order = serial-chain order
        _cache["tip"] = sim.getObject(TIP)
    return _cache


# ---------------------------------------------------------------- joints
def get_joint_angles(sim):
    h = _handles(sim)
    return np.array([sim.getJointPosition(j) for j in h["joints"]])


def set_joint_angles(sim, q):
    """Command joint angles (rad). Kinematic move, so it is deterministic for the demo."""
    h = _handles(sim)
    q = np.asarray(q, dtype=float)
    if q.shape != (6,):
        raise ValueError("q must have shape (6,)")
    for j, v in zip(h["joints"], q):
        sim.setJointPosition(j, float(v))


# ---------------------------------------------------------------- poses
def get_tip_pose(sim):
    h = _handles(sim)
    m = np.eye(4)
    m[:3, :] = np.array(sim.getObjectMatrix(h["tip"], h["base"])).reshape(3, 4)
    return m


def get_object_position(sim, name=PART):
    h = _handles(sim)
    return np.array(sim.getObjectPosition(sim.getObject(name), h["base"]))


# ---------------------------------------------------------------- gripper
def close_gripper(sim):
    """Attach the part to the tool if within GRIP_RANGE. Returns True on success."""
    h = _handles(sim)
    part = sim.getObject(PART)
    d = np.linalg.norm(np.array(sim.getObjectPosition(part, h["tip"])))
    if d > GRIP_RANGE:
        return False
    sim.setObjectParent(part, h["tip"], True)   # True = keep world pose
    return True


def open_gripper(sim):
    h = _handles(sim)
    part = sim.getObject(PART)
    if sim.getObjectParent(part) == h["tip"]:
        sim.setObjectParent(part, -1, True)
    return True


def part_in_bin(sim, bin_name="/Bin", tol=0.12):
    """True if the part ended up over the bin (success check for experiments)."""
    h = _handles(sim)
    b = np.array(sim.getObjectPosition(sim.getObject(bin_name), h["base"]))
    p = get_object_position(sim)
    return bool(np.linalg.norm(p[:2] - b[:2]) < tol and p[2] < 0.15)


def reset_scene_objects(sim, part_xyz=PART_START):
    """Release the part and put it back at a start position (base frame)."""
    h = _handles(sim)
    open_gripper(sim)
    sim.setObjectPosition(sim.getObject(PART), h["base"], [float(v) for v in part_xyz])


# ---------------------------------------------------------------- validation
def verify_fk(sim, n=5, seed=0):
    """Compare dh_fk with the simulator for n random joint sets.

    Returns the max position error in metres (objective: D-H model matches the sim).
    """
    from dh_fk import forward_kinematics
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n):
        q = rng.uniform(-1.5, 1.5, 6)
        set_joint_angles(sim, q)
        errs.append(np.linalg.norm(get_tip_pose(sim)[:3, 3] - forward_kinematics(q)[:3, 3]))
    return max(errs)


if __name__ == "__main__":
    client, sim = connect()
    print("Connected to CoppeliaSim")
    start_simulation(client, sim)
    print("joints:", np.round(get_joint_angles(sim), 3))
    print("tip pos:", np.round(get_tip_pose(sim)[:3, 3], 3))
    print("max FK error (m):", verify_fk(sim))
    stop_simulation(client, sim)
