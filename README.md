
# Stochastic Utility-Aware Multi-Robot Task Allocation with Topometric CBS

**From-scratch Python implementation of a unified MRTA–MAPF–Estimation framework**
combining stochastic utility-aware task allocation (SCoBA-inspired),
conflict-free multi-agent path finding using topometric PM-CBS,
and uncertainty-aware continuous execution with EKF localization.

---

## Overview

This repository implements an end-to-end multi-robot coordination system for
warehouse-like environments, where multiple mobile robots must execute
time-constrained service tasks safely and efficiently in a shared workspace.

The core challenge addressed is the tight coupling of three traditionally
separate problems:

- Multi-Robot Task Allocation (MRTA) under deadlines, travel costs, and
  heterogeneous utilities.
- Multi-Agent Path Finding (MAPF) with strict collision-avoidance guarantees
  in narrow, shared regions.
- Execution under uncertainty, where robots follow planned trajectories using
  noisy sensing and localization.

To address this, we implement a principled pipeline:

[ SCoBA Allocation → Topometric PM-CBS → Grid A* Refinement → Continuous Execution + EKF ]

The system is fully self-contained, uses no ROS/Gazebo dependencies, and is
designed to support fair comparisons against greedy and learning-based baselines.

---

## Environment

The workspace is a 2D warehouse represented as an occupancy grid with obstacles
(shelves, walls) and free corridors.

Robots operate in continuous space but plan over an inflated grid to ensure
safety margins.

Each free grid cell (x, y) corresponds to a continuous position:

[ p = (x + 0.5, y + 0.5)^T ]

### Environment Map

<p align="center">



</p>

---

## Robot Model and Dynamics

Each robot follows a unicycle kinematic model.

**State**
[ x_i = (x_i, y_i, θ_i) ]

**Control**
[ u_i = (v_i, ω_i) ]

**Discrete-Time Dynamics**

[
x_i(t+1) = x_i(t) + v_i(t) · cos(θ_i(t)) · Δt  
y_i(t+1) = y_i(t) + v_i(t) · sin(θ_i(t)) · Δt  
θ_i(t+1) = θ_i(t) + ω_i(t) · Δt
]

Velocity and angular rates are bounded. Collisions with obstacles are detected
using the inflated grid; upon collision, robots revert to their previous pose.

---

## Task Model

The environment issues a set of service tasks:

[ T = {T_1, T_2, …, T_M} ]

Each task is defined as:

[ T_j = (p_j, r_j, d_j, Δ_j, u_j) ]

where:
- p_j is the task location,
- [r_j, d_j] is a hard time window,
- Δ_j is the service duration,
- u_j is the task utility.

A task contributes utility at most once, even if multiple robots attempt it.

---

## Stochastic Utility-Aware Allocation (SCoBA-Inspired)

Task allocation is performed using a stochastic, utility-aware procedure inspired
by the SCoBA framework.

### Travel Time Estimation

Travel time between locations is estimated using grid-based A* search:

[ τ(p_a, p_b) ]

### Success Probability

Each task is assigned a bounded success probability p_j in [0, 1], which
decreases with late arrival relative to the deadline and excessive travel time.

### Expected Utility

[ E[U_j] = p_j · u_j ]

Each robot constructs a feasible task sequence that maximizes cumulative
expected utility subject to time-window constraints.

To avoid duplicate assignments, a CBS-style branching mechanism is applied at
the allocation level: when two robots select the same task, two branches are
created, forbidding that task for one robot in each branch.

---

## Topometric Representation

To enable scalable multi-agent planning, the inflated grid is decomposed into
connected free-space regions:

[ R = {R_1, R_2, …, R_K} ]

Adjacency between regions defines a topometric graph:

[ G_topo = (R, E) ]

Each region is treated as a shared resource that cannot be occupied by multiple
robots during overlapping time intervals.

### Topometric Graph Visualization

<p align="center">



</p>

---

## Priority-Based Multi-Agent CBS (PM-CBS)

Given robot start regions and goal regions (from allocated tasks), planning is
performed using Priority-Based Multi-Agent Conflict-Based Search (PM-CBS) at
the region-time level.

### Region-Time Plans

For robot i, a plan is a sequence of tuples:

[ (R_i1, t_i1_in, t_i1_out), (R_i2, t_i2_in, t_i2_out), … ]

### Conflict Definition

A conflict occurs if two robots occupy the same region during overlapping time
intervals:

[ [t_i_k_in, t_i_k_out] ∩ [t_j_l_in, t_j_l_out] ≠ ∅ ]

### CBS Branching

When a conflict is detected, two branches are generated:
- forbid the conflicting region-time interval for robot i, or
- forbid it for robot j.

Only the affected robot is replanned in each branch, ensuring collision-free
coordination at the region level.

---

## Grid-Level Path Refinement

Each region-level plan is refined into a dense, executable path using grid A*
search between representative cells of consecutive regions.

Waypoints are generated as continuous positions:

[ w_k = (x_k + 0.5, y_k + 0.5) ]

This stage ensures geometric feasibility without reintroducing inter-robot
conflicts.

---

## Continuous Execution and Localization

Robots follow waypoints using a proportional controller while maintaining a
belief over their state using an Extended Kalman Filter (EKF).

### EKF Prediction

[ μ⁻ = f(μ, u),   P⁻ = F · P · F^T + Q ]

### Measurement Updates

- Noisy GPS-like position updates
- Selective LiDAR range updates using a conservative beam model

LiDAR is used for state correction and validation, not for replanning.

---

## Baseline Policies

### Greedy A*

- Each robot repeatedly selects the nearest unfinished task.
- Paths are planned independently using grid A*.
- No coordination or collision avoidance between robots.

### Reinforcement Learning Baseline

- Tabular Q-learning over tasks.
- Sparse rewards and no safety constraints.
- Included as a learning-based comparison baseline.

---

## Qualitative Results

### Greedy A* Simulation

<p align="center">



</p>

### SCoBA + PM-CBS Simulation

<p align="center">



</p>

---

## Evaluation Metrics

We report the following metrics:

- Tasks Completed:  
  [ C = |T_succ| ]

- Total Utility:  
  [ U_tot = sum_{j in T_succ} u_j ]

- Total Distance Travelled:  
  [ D = sum_i ∫ |v_i(t)| dt ]

- Utility Efficiency:  
  [ η_d = U_tot / D,   η_s = U_tot / steps ]

- Collision Count

---

## Repository Structure

```
.
├── agents_and_tasks.py
├── scoba_allocation.py
├── cbs_topometric.py
├── low_level_topometric_planner.py
├── map_warehouse.py
├── main_simulation.py
├── eval_policies.py
└── README.md
```

---

## References

- Scott Fredriksson, Yifan Bai, Akshit Saradagi, and George Nikolakopoulos.  
  Multi-Agent Path Finding Using Conflict-Based Search and Structural-Semantic
  Topometric Maps. arXiv:2501.17661, 2025.

- Shushman Choudhury, Jayesh K. Gupta, Mykel J. Kochenderfer, Dorsa Sadigh, and
  Jeannette Bohg.  
  Dynamic Multi-Robot Task Allocation under Uncertainty and Temporal
  Constraints. arXiv:2005.13109, 2020.

---

Research / educational use only.
