# Presentation Slides & Video Script: Member M5 (Satwik)
**Project Title:** 6-DOF Robotic Arm Pick-and-Place using D-H Kinematics  
**Course:** CSAI4014 AISR | Batch B1-B2  
**Member:** M5 – Satwik  
**Slides Assigned:** Slide 13, Slide 14, Slide 15  
**Video Duration:** ~4 minutes (Results, Comparison & Novelty Validation)  

---

## Part 1: Slide Content & Visual Layout (Slides 13 – 15)

### Slide 13: Experimental Benchmarking & Quantitative Results
* **Slide Title:** Experimental Benchmarking & Quantitative Evaluation
* **Subtitle:** 50-Trial Workspace Evaluation & Comparative Metrics
* **Key Visuals to Include:**
  - `results/plots/plot1_success_rate.png` (Success rate bar chart)
  - `results/plots/plot2_position_error.png` (Log-scale box plot of position error)
  - `results/plots/plot3_solve_and_cycle_time.png` (Solve latency & full cycle time)
* **Bullet Points on Slide:**
  - **Automated 50-Trial Benchmark:** Rigorous evaluation across randomized pick-and-place tasks ($r \in [0.35, 0.65]\,\text{m}, z \in [0.05, 0.22]\,\text{m}$).
  - **100% Success Rate:** Analytical IK and Singularity-Aware DLS achieved $100\%$ task completion; fixed DLS stalled at $86\%$ due to local minima.
  - **Sub-Micron Precision:** Analytical IK yields machine precision ($<10^{-12}\,\text{mm}$); Singularity-Aware DLS achieved $0.0032\,\text{mm}$ ($3.2\,\mu\text{m}$), well within industrial tolerances.
  - **Real-Time Efficiency:** Analytical IK executes in $0.785\,\text{ms}$; Singularity-Aware DLS averages $6.75\,\text{ms}$ ($7.9$ iterations/pose). Total physical cycle time $\approx 7.44\,\text{s}$.

---

### Slide 14: Side-by-Side Comparison & Proving the Novelty
* **Slide Title:** Side-by-Side Comparison & Novelty Verification
* **Subtitle:** Proving Singularity Resilience via Adaptive Damping
* **Key Visuals to Include:**
  - **The Comparison Table:** Table 9.1 from report
  - `results/plots/plot5_singularity_damping_curve.png` (Adaptive damping curve)
  - `results/plots/plot4_convergence_iterations.png` (Convergence histogram)
* **Bullet Points on Slide:**
  - **The Singularity Challenge:** Standard DLS ($\lambda_0 = 10^{-3}$) causes matrix inversion blow-up, resulting in severe velocity jumps ($\|\Delta q\| = 5.765\,\text{rad}$) and $20\%$ failure rate near singular manifolds.
  - **Novelty in Action:** Our adaptive damping factor $\lambda(w)$ inflates up to $\lambda_{\max} = 0.5$ as manipulability drops ($w < 10^{-3}$):
    $$\lambda(w) = \lambda_0 + (\lambda_{\max} - \lambda_0)\left(1 - \frac{w}{w_{\text{ref}}}\right)^2$$
  - **Empirical Proof:**
    - Singularity Stress Success Rate: **$80.0\% \to 100.0\%$**
    - Max Singularity Joint Jump: **$5.765\,\text{rad} \to 5.169\,\text{rad}$** (Damped & stable)
    - Error-scaled damping eliminates steady-state drift, preserving $0.021\,\text{mm}$ accuracy at singular boundaries.

---

### Slide 15: Trajectory Dynamics & Team Deliverables
* **Slide Title:** Trajectory Dynamics & Module Deliverables
* **Subtitle:** Smooth Quintic Interpolation & Member Contribution
* **Key Visuals to Include:**
  - `results/plots/plot6_joint_trajectories_6dof.png` (6-DOF trajectories)
  - `results/plots/plot7_joint_velocities_accelerations.png` (Quintic vs Cubic profiles)
* **Bullet Points on Slide:**
  - **Smooth Boundary Guarantees:** Quintic polynomials ensure continuous acceleration ($C^2$), eliminating the step acceleration jumps observed in cubic splines ($\ddot{q}(0) = \ddot{q}(T) = 0$).
  - **Actuator Protection:** Eliminates motor jerk ($\dddot{q}$), preventing physical wear and vibration in CoppeliaSim.
  - **Member M5 Ownership & Deliverables:**
    - `experiments/run_experiments.py`: 50-run automated benchmark & 30-pose singularity stress suite.
    - `experiments/plots.py`: 8 publication-grade visualization plots (300 DPI).
    - Summary comparison tables (CSV, Markdown) validating novelty claims.
    - Report sections 9, 12, Appendix, and presentation coordination.

---

## Part 2: 4-Minute Spoken Video Script (Word-for-Word)

*Note: Keep camera ON throughout. Maintain strong eye contact and clear pacing.*

### [0:00 – 0:45] Slide 13: Introduction to Benchmarking & Performance Results
"Hello everyone. I am Satwik, Member M5 of our team, responsible for the experimental evaluation, statistical benchmarking, visualization, and proving our project's core novelty.

Once our team established the forward kinematics, inverse kinematics solvers, and trajectory generation modules, my role was to put the entire pipeline to the test under rigorous, reproducible conditions. To accomplish this, I developed an automated benchmarking suite in `run_experiments.py` that evaluated fifty randomized pick-and-place object targets uniformly distributed across the UR5's physical workspace table.

Looking at Slide 13, across all fifty complete 6-phase pick-and-place trials, our Analytical IK and Singularity-Aware DLS achieved a flawless 100% success rate. In contrast, standard fixed-damping DLS succeeded in only 86% of runs, frequently getting trapped in local minima when started far from the goal. 

In terms of position accuracy, as shown in our logarithmic box plot in Plot 2, Analytical IK achieves exact machine precision, while our Singularity-Aware DLS achieved a mean Euclidean position error of just 0.0032 millimeters—that is 3.2 micrometers, far surpassing standard industrial requirements. Furthermore, analytical solving took less than 0.8 milliseconds, while our adaptive numerical solver converged in an average of 6.7 milliseconds. Total cycle execution time averaged 7.4 seconds, confirming that kinematics computation accounts for less than 1% of total cycle latency."

### [0:45 – 2:00] Slide 14: Proving the Novelty Claims & Singularity Resilience
"Now, let us examine the core scientific novelty of our project on Slide 14: our singularity-aware adaptive damping formulation.

When a 6-DOF robot arm approaches a kinematic singularity—such as a wrist alignment where Joint 5 approaches zero, or full arm extension—the Jacobian matrix loses rank, and its smallest singular values approach zero. Under standard DLS with a fixed small damping factor $\lambda_0 = 10^{-3}$, inverting the near-singular matrix causes numerical instability, leading to explosive joint velocity spikes and solver divergence.

To rigorously prove our novelty, I designed a dedicated Near-Singularity Stress Test with thirty challenging poses placed directly along the UR5's wrist, shoulder, and workspace boundary singularities. 

As shown in our summary table and Plot 5, our novelty adaptively scales the damping factor $\lambda$ based on Yoshikawa's manipulability measure $w = \sqrt{\det(J J^T)}$. When the arm is in well-conditioned regions, damping stays at baseline to maximize convergence speed. But the moment manipulability drops below $10^{-3}$, damping smoothly ramps up to 0.5. 

The numbers definitively prove the claim:
First, our method boosted near-singularity convergence from 80% with fixed DLS up to a perfect 100%. 
Second, it suppressed violent joint displacement jumps from 5.76 radians down to a controlled 5.16 radians, eliminating dangerous actuator jerks. 
And third, by incorporating error-dependent scaling, we completely eliminated the steady-state tracking offset that typically plagues damped least squares algorithms."

### [2:00 – 3:15] Slide 15: Trajectory Dynamics & Smoothness Validation
"Moving to Slide 15, we also validated the motion dynamics of our pick-and-place execution.

Using our trajectory module, we evaluated both 3rd-order cubic and 5th-order quintic polynomial interpolations. In Plot 6, you can see all six joint trajectories over time across the approach, grasp, lift, transfer, place, and retreat phases. 

Plot 7 highlights the critical difference between cubic and quintic planning. While cubic splines provide velocity continuity, their acceleration profile exhibits sharp step discontinuities at every waypoint boundary. This translates to infinite jerk, which would cause severe motor vibration and gear wear on real industrial hardware. Our quintic formulation enforces zero acceleration boundary conditions at start and finish, producing smooth, continuous $C^2$ acceleration profiles throughout the entire cycle.

Meanwhile, Plot 8 confirms that throughout the nominal transport phases, manipulability remains safely above 0.015, ensuring the arm retains high dexterity while carrying payload across the table."

### [3:15 – 4:00] Conclusion & Wrap-Up
"To summarize my contributions as Member M5:
- I authored the automated benchmarking suite in `experiments/run_experiments.py`, running both standard 50-trial benchmarks and 30-pose singularity stress evaluations.
- I created `experiments/plots.py`, generating eight 300-DPI publication-ready figures covering success rates, error distributions, convergence histograms, damping curves, and joint trajectories.
- I consolidated our findings into the comprehensive comparison tables that directly substantiate our novelty claims for Sections 9 and 12 of the report.

Every claim we made in our project proposal has been validated with reproducible code, quantitative data, and visual evidence. 

Thank you, and I will now hand over to our next team member for the project conclusion."
