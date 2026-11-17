"""
M4: Pick-and-place state machine with trajectory execution and IK method selection.

State Machine Flow:
1. STANDBY: Move arm to default home posture.
2. APPROACH_PICK: Move end-effector 10 cm directly above the target object.
3. DESCEND_GRASP: Move down to object position and engage gripper.
4. LIFT: Lift object 15 cm vertically off table.
5. MOVE_PLACE: Transfer object above target drop bin.
6. DESCEND_PLACE: Lower object to target container and release gripper.
7. RETREAT: Lift end-effector clear and return to STANDBY.

Supports:
- Command-line switch: --method analytical | dls | adaptive
- Simulator mode (via CoppeliaSim ZeroMQ API) or Standalone verification mode.
"""
import os
import sys
import time
import argparse
import numpy as np

# Ensure src in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dh_fk as fk
import ik_analytical as ana
import ik_dls as dls
import trajectory as traj


def rot_downward(yaw=0.0):
    """Gripper orientation facing downward with azimuthal yaw."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Rx_pi = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return Rz @ Rx_pi


def make_pose(position, orientation):
    T = np.eye(4)
    T[:3, :3] = orientation
    T[:3, 3] = position
    return T


class PickAndPlaceController:
    def __init__(self, method="analytical", use_sim=True):
        self.method = method
        self.use_sim = use_sim
        self.sim = None
        self.client = None
        self.joint_handles = []
        self.standby_q = np.array([0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0])
        self.current_q = self.standby_q.copy()
        
        if self.use_sim:
            try:
                from zmq_client import connect
                self.client, self.sim = connect()
                print("[SIM] Connected to CoppeliaSim ZeroMQ Remote API.")
                self._init_sim_handles()
            except Exception as e:
                print(f"[SIM] CoppeliaSim not reachable ({e}). Falling back to standalone kinematics mode.")
                self.use_sim = False
                
    def _init_sim_handles(self):
        """Retrieve joint handles from CoppeliaSim."""
        try:
            self.joint_handles = [
                self.sim.getObject(f"/UR5/joint{i}") for i in range(1, 7)
            ]
        except Exception:
            try:
                # Alternate naming convention in some UR5 scenes
                self.joint_handles = [
                    self.sim.getObject(f"/UR5_joint{i}") for i in range(1, 7)
                ]
            except Exception as e:
                print(f"[SIM Warning] Could not resolve joint handles ({e}).")

    def solve_ik(self, T_target):
        """Solve inverse kinematics using selected method."""
        if self.method == "analytical":
            return ana.inverse_kinematics(T_target, q_prefer=self.current_q)
        elif self.method == "dls":
            return dls.inverse_kinematics(
                T_target, q0=self.current_q,
                lam0=1e-3, lam_max=1e-3,
                max_iter=200, pos_tol=1e-4, rot_tol=1e-4,
                multi_start=False
            )
        elif self.method == "adaptive":
            return dls.inverse_kinematics(
                T_target, q0=self.current_q,
                lam0=1e-3, lam_max=0.5, w_ref=1e-3,
                max_iter=200, pos_tol=1e-4, rot_tol=1e-4,
                multi_start=True
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def execute_trajectory(self, q_goal, duration=2.0, dt=0.02, phase_name=""):
        """Execute smooth quintic joint trajectory from current_q to q_goal."""
        times, q_traj, qd_traj, qdd_traj = traj.generate_trajectory(
            self.current_q, q_goal, duration=duration, dt=dt, method="quintic"
        )
        print(f"  -> Executing {phase_name}: duration={duration:.1f}s, steps={len(times)}")
        
        for i, q in enumerate(q_traj):
            if self.use_sim and self.sim and self.joint_handles:
                for jh, val in zip(self.joint_handles, q):
                    self.sim.setJointTargetPosition(jh, float(val))
                time.sleep(dt)
            self.current_q = q
            
        self.current_q = q_goal.copy()

    def set_gripper(self, close=True):
        """Simulate or trigger gripper action."""
        action = "CLOSED (Grasped)" if close else "OPENED (Released)"
        print(f"  [GRIPPER] {action}")
        time.sleep(0.3)

    def run_cycle(self, pick_pos, place_pos):
        """Execute full 7-stage pick-and-place sequence."""
        print(f"\n=======================================================")
        print(f"Starting Pick-and-Place Cycle | Method: {self.method.upper()}")
        print(f"Pick Location:  {pick_pos}")
        print(f"Place Location: {place_pos}")
        print(f"=======================================================")
        
        R_pick = rot_downward(np.arctan2(pick_pos[1], pick_pos[0]))
        R_place = rot_downward(np.arctan2(place_pos[1], place_pos[0]))
        
        phases = [
            ("1. STANDBY", fk.forward_kinematics(self.standby_q), 1.5, False),
            ("2. APPROACH PICK", make_pose(pick_pos + np.array([0, 0, 0.10]), R_pick), 1.5, False),
            ("3. DESCEND GRASP", make_pose(pick_pos, R_pick), 0.8, True),
            ("4. LIFT OBJECT", make_pose(pick_pos + np.array([0, 0, 0.15]), R_pick), 0.8, True),
            ("5. MOVE TO PLACE", make_pose(place_pos + np.array([0, 0, 0.15]), R_place), 2.0, True),
            ("6. DESCEND PLACE", make_pose(place_pos, R_place), 0.8, False),
            ("7. RETREAT TO STANDBY", fk.forward_kinematics(self.standby_q), 1.5, False),
        ]
        
        start_time = time.time()
        for name, T_target, dur, grip_state in phases:
            t0 = time.perf_counter()
            q_target = self.solve_ik(T_target)
            ik_time = (time.perf_counter() - t0) * 1000.0
            
            p_actual = fk.forward_kinematics(q_target)[:3, 3]
            pos_err = np.linalg.norm(T_target[:3, 3] - p_actual) * 1000.0
            print(f"[{name}] IK solved in {ik_time:.2f} ms | Pos error: {pos_err:.4f} mm")
            
            self.execute_trajectory(q_target, duration=dur, phase_name=name)
            if "GRASP" in name:
                self.set_gripper(close=True)
            elif "DESCEND PLACE" in name:
                self.set_gripper(close=False)
                
        elapsed = time.time() - start_time
        print(f"\n[SUCCESS] Pick-and-place cycle completed successfully in {elapsed:.2f} seconds.")


def main():
    parser = argparse.ArgumentParser(description="UR5 Pick-and-Place Controller")
    parser.add_argument("--method", choices=["analytical", "dls", "adaptive"], default="adaptive",
                        help="IK Method: analytical, dls (fixed), or adaptive (singularity-aware)")
    parser.add_argument("--no-sim", action="store_true", help="Run standalone kinematics without simulator")
    parser.add_argument("--pick", nargs=3, type=float, default=[0.45, 0.20, 0.10], help="Pick [x, y, z] in meters")
    parser.add_argument("--place", nargs=3, type=float, default=[-0.35, 0.35, 0.08], help="Place [x, y, z] in meters")
    args = parser.parse_args()

    controller = PickAndPlaceController(method=args.method, use_sim=not args.no_sim)
    controller.run_cycle(pick_pos=np.array(args.pick), place_pos=np.array(args.place))


if __name__ == "__main__":
    main()
