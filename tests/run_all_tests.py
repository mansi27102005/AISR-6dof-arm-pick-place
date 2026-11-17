"""
Master test runner for the entire 6-DOF UR5 Pick-and-Place project.
Runs:
1. tests/test_dh_fk.py (D-H parameters, FK, Jacobians, singularities)
2. tests/test_ik.py (Analytical & Numerical DLS IK solvers)
3. tests/test_trajectory.py (Quintic and Cubic joint trajectories)

Usage:
    python tests/run_all_tests.py
"""
import os
import sys
import time
import subprocess

TESTS = [
    ("Forward Kinematics & D-H Table (M2)", "tests/test_dh_fk.py"),
    ("Inverse Kinematics & Novelty DLS (M3)", "tests/test_ik.py"),
    ("Trajectory Generation (M4)", "tests/test_trajectory.py"),
]


def run():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 65)
    print("  RUNNING COMPLETE VERIFICATION TEST SUITE (UR5 PICK & PLACE)")
    print("=" * 65)
    
    total_passed = 0
    total_failed = 0
    start_total = time.time()
    
    for title, rel_path in TESTS:
        full_path = os.path.join(repo_root, rel_path)
        print(f"\n[SUITE] {title}")
        print(f"        Path: {rel_path}")
        t0 = time.time()
        res = subprocess.run([sys.executable, full_path], cwd=repo_root, capture_output=True, text=True)
        dur = time.time() - t0
        
        if res.returncode == 0:
            print(f"        STATUS: PASSED ({dur:.2f} s)")
            total_passed += 1
            for line in res.stdout.strip().split("\n"):
                if "PASS" in line or "passed" in line or "solved" in line:
                    print(f"          {line}")
        else:
            print(f"        STATUS: FAILED ({dur:.2f} s)")
            total_failed += 1
            print("        Output / Error:")
            print(res.stderr or res.stdout)
            
    total_time = time.time() - start_total
    print("\n" + "=" * 65)
    print(f"VERIFICATION SUMMARY: {total_passed} Passed | {total_failed} Failed | Time: {total_time:.2f}s")
    print("=" * 65)
    
    return total_failed == 0


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)
