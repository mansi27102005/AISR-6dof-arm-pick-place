"""Build the pick-and-place scene in CoppeliaSim and save it as scenes/pick_place.ttt.

Usage (CoppeliaSim must be running):
    python scenes/build_scene.py

If the UR5 model is not found, set COPPELIASIM_ROOT_DIR to the CoppeliaSim folder.

Layout (robot base frame = world frame, metres):
    UR5 base            (0, 0, 0)
    conveyor            (0.40, 0.20), top at z = 0.02, the part starts on it
    bin                 (0.40, -0.20)
These match the defaults in src/pick_place.py (--obj 0.4 0.2 0.05, --place 0.4 -0.2 0.05).
"""
import os
import sys
from pathlib import Path

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

ROOT = Path(__file__).resolve().parent
UR5_MODEL = "models/robots/non-mobile/UR5.ttm"


def find_model():
    cands = [os.environ.get("COPPELIASIM_ROOT_DIR", "")]
    for base in ("C:/Program Files/CoppeliaRobotics", "C:/Program Files (x86)/CoppeliaRobotics"):
        if os.path.isdir(base):
            cands += [os.path.join(base, d) for d in os.listdir(base)]
    for c in cands:
        p = os.path.join(c, UR5_MODEL)
        if c and os.path.isfile(p):
            return p
    sys.exit("UR5.ttm not found. Set COPPELIASIM_ROOT_DIR to the CoppeliaSim folder.")


def box(sim, size, pos, color, name):
    h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size, 0)
    sim.setObjectPosition(h, -1, pos)
    sim.setObjectAlias(h, name)
    sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, color)
    sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
    return h


def main():
    client = RemoteAPIClient()
    sim = client.require("sim")
    sim.closeScene()

    robot = sim.loadModel(find_model())
    sim.setObjectAlias(robot, "UR5")
    sim.setObjectPosition(robot, -1, [0, 0, 0])
    # drop the model's own control script so joints are driven only from Python
    scr = sim.getScript(sim.scripttype_childscript, robot)
    if scr != -1:
        sim.removeScript(scr)

    # tool-tip dummy: on the model's last dummy (connection) if present, else the last joint
    dummies = sim.getObjectsInTree(robot, sim.object_dummy_type, 0)
    joints = sim.getObjectsInTree(robot, sim.object_joint_type, 0)
    parent = dummies[-1] if dummies else joints[-1]
    tip = sim.createDummy(0.02)
    sim.setObjectAlias(tip, "GripperTip")
    sim.setObjectParent(tip, parent, False)
    sim.setObjectPosition(tip, parent, [0, 0, 0])
    sim.setObjectOrientation(tip, parent, [0, 0, 0])

    # conveyor, part and bin
    box(sim, [0.30, 0.30, 0.02], [0.40, 0.20, 0.01], [0.2, 0.2, 0.2], "Conveyor")
    box(sim, [0.045, 0.045, 0.045], [0.40, 0.20, 0.0425], [0.9, 0.3, 0.1], "Part")
    bx, by = 0.40, -0.20
    box(sim, [0.25, 0.25, 0.01], [bx, by, 0.005], [0.2, 0.4, 0.8], "Bin")  # bin floor
    for dx, dy, sx, sy in [(0.125, 0, 0.01, 0.25), (-0.125, 0, 0.01, 0.25),
                           (0, 0.125, 0.25, 0.01), (0, -0.125, 0.25, 0.01)]:
        box(sim, [sx, sy, 0.06], [bx + dx, by + dy, 0.03], [0.2, 0.4, 0.8], "BinWall")

    out = str(ROOT / "pick_place.ttt")
    sim.saveScene(out)
    print("Scene saved to", out)


if __name__ == "__main__":
    main()
