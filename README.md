# 6-DOF Robotic Arm Pick-and-Place using D-H Kinematics

A simulated 6-DOF robot arm (UR5 / IRB 140) that picks up an object and places it at a target location. The arm is modelled with Denavit-Hartenberg (D-H) kinematics, controlled from Python through CoppeliaSim's ZeroMQ remote API, and tested with two inverse kinematics methods.

**Course:** CSAI4014 Algorithms for Intelligent Systems and Robotics
**Programme:** B.Tech CSE (AI & ML), Semester VII, UPES Dehradun
**Batch:** B1-B2  |  **Project No.:** `__`  |  **Group No.:** `__`

---

## What this project does

1. Builds the D-H table of the arm and computes forward kinematics.
2. Solves inverse kinematics two ways and compares them:
   - analytical (closed form)
   - numerical (Jacobian with damped least squares)
3. Plans smooth joint trajectories between poses.
4. Runs a full pick-and-place cycle in CoppeliaSim: approach, grasp, lift, move, place, retreat.
5. Measures position error, success rate and cycle time over many random object positions.

## Objectives

- Model the arm with a verified D-H table (FK matches the simulator pose).
- Reach target poses with position error below `__` mm.
- Complete at least `__`% of pick-and-place runs successfully.
- Compare two IK methods on the same set of test positions.

## Tech stack

| Part | Tool |
|---|---|
| Simulator | CoppeliaSim (Edu) |
| Robot | UR5 / IRB 140 |
| Language | Python 3.10+ |
| Simulator link | ZeroMQ remote API |
| Maths and plots | NumPy, SciPy, Matplotlib, Pandas |

## Repository structure

```
.
├── src/                    main code, one file per module
│   ├── zmq_client.py       connection to CoppeliaSim
│   ├── dh_fk.py            D-H table and forward kinematics
│   ├── ik_analytical.py    closed-form inverse kinematics
│   ├── ik_dls.py           numerical IK (damped least squares)
│   ├── trajectory.py       joint-space trajectory generation
│   └── pick_place.py       pick-and-place state machine (main entry point)
├── experiments/            scripts that generate the results
│   ├── run_experiments.py
│   └── plots.py
├── scenes/                 CoppeliaSim scene files (.ttt)
├── results/                CSV logs and plots
├── docs/                   meeting logs and notes
├── report/                 report drafts
├── requirements.txt
└── README.md
```

## How to run

1. Install CoppeliaSim (Edu) and open the scene file in `scenes/`.
2. Install the Python packages:
   ```
   pip install -r requirements.txt
   ```
3. Press **Play** in CoppeliaSim.
4. Run the pick-and-place demo:
   ```
   python src/pick_place.py
   ```
5. To reproduce the results and plots:
   ```
   python experiments/run_experiments.py
   python experiments/plots.py
   ```

## Results

*To be filled in after the experiments.*

| Metric | Analytical IK | Damped least squares IK |
|---|---|---|
| Success rate | | |
| Mean position error (mm) | | |
| Mean cycle time (s) | | |

## Team

| Member | SAP ID | GitHub | Module |
|---|---|---|---|
| M1 | | | Scene setup and ZeroMQ connection |
| M2 | | | D-H table and forward kinematics |
| M3 | | | Inverse kinematics |
| M4 | | | Trajectory planning and pick-and-place logic |
| M5 | | | Experiments, plots and results |

## Working rules

- Everyone commits from their own GitHub account.
- Each member works on their own branch (for example `m2-forward-kinematics`) and merges to `main` through a pull request.
- Commit messages say what changed, like `add damped least squares IK`.
- Keep each module in its own file, so contributions are easy to trace.

## Links

- Demo video (YouTube, unlisted): `__`
- Meeting log: `docs/meeting_logs/meeting_log.md`

## References

*Added as the report is written.*
