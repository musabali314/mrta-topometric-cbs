"""
main_simulation.py
------------------

Continuous warehouse simulation with advanced visualization:

  - True robot poses (solid colored path)
  - EKF estimated poses (black dashed path)
  - Heading arrows (true + estimated)
  - Lidar ray visualization
  - PM-CBS conflict region visualization

Policies supported:
  - scoba_cbs    : SCoBA allocation + PM-CBS coordination  (main policy)
  - greedy_astar : baseline (placeholder hooks)
  - rl1          : simple RL baseline (placeholder hooks)
"""

import heapq
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from map_warehouse import (
    create_random_warehouse,
    build_topometric_map,
    safe_free_cell_with_clearance,
    inflate_obstacles,
)
from agents_and_tasks import Robot, Task
from scoba_allocation import SCoBASolver
from low_level_topometric_planner import travel_time_from_cells
from cbs_topometric import PMCBS


# =====================================================================
#   A* GRID HELPERS
# =====================================================================

def grid_neighbors(grid, x, y):
    """4-connected neighbors in FREE space."""
    h, w = grid.shape
    for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
        if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0:
            yield nx, ny


def grid_astar(grid, start, goal):
    """
    Basic A* from start=(sx,sy) to goal=(gx,gy) in grid cell coordinates.
    Returns a list of (x,y) cells, or [] if no path.
    """
    sx, sy = start
    gx, gy = goal
    if start == goal:
        return [start]

    open_set = []
    heapq.heappush(open_set, (0.0, sx, sy))
    came_from = {(sx, sy): None}
    g_cost = {(sx, sy): 0.0}

    def h(a, b):
        # Manhattan heuristic
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while open_set:
        _, x, y = heapq.heappop(open_set)

        if (x, y) == (gx, gy):
            # reconstruct
            path = []
            cur = (gx, gy)
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            return path[::-1]

        for nx, ny in grid_neighbors(grid, x, y):
            newg = g_cost[(x, y)] + 1.0
            if (nx, ny) not in g_cost or newg < g_cost[(nx, ny)]:
                g_cost[(nx, ny)] = newg
                came_from[(nx, ny)] = (x, y)
                f = newg + h((nx, ny), (gx, gy))
                heapq.heappush(open_set, (f, nx, ny))

    return []


def agent_color(i):
    cols = ["red", "blue", "green", "orange", "purple", "cyan"]
    return cols[i % len(cols)]


def draw_tasks(ax, tasks):
    """Readable task markers + labels."""
    for t in tasks:
        x, y = t.location
        px, py = x + 0.5, y + 0.5

        ax.scatter(
            px,
            py,
            marker="*",
            s=260,
            color="gold",
            edgecolors="black",
            zorder=5,
        )
        ax.text(
            px,
            py - 0.45,
            t.name,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="black",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            zorder=6,
        )


# =====================================================================
#   BUILD SCENARIO
# =====================================================================

def build_scenario(num_agents=3, num_tasks=7, seed=42):
    """
    Returns:
        grid_render, grid_plan, topo, robots, tasks, cell_to_region, region_to_cell
    """
    if seed is not None:
        random.seed(seed)
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # 1) Warehouse
    grid_render = create_random_warehouse(seed=seed)
    # 2) Inflated map for planning
    grid_plan = inflate_obstacles(grid_render, inflation_radius=1)

    # 3) Topometric
    topo = build_topometric_map(grid_plan)
    cell_to_region = topo.cell_to_region
    region_to_cell = topo.region_to_cell

    # 4) Robots
    robots = []
    for i in range(num_agents):
        cx, cy = safe_free_cell_with_clearance(grid_plan, rng, clearance=1)
        init_pose = (cx + 0.5, cy + 0.5, 0.0)
        r = Robot(f"A{i}", init_pose)
        r.start_location = (cx, cy)
        robots.append(r)

    # 5) Tasks (realistic utilities)
    tasks = []

    # Utility weights (tune as needed)
    w_p = 5.0    # priority multiplier
    w_v = 1.0    # base value multiplier
    w_d = 30.0   # urgency multiplier

    for i in range(num_tasks):
        tx, ty = safe_free_cell_with_clearance(grid_plan, rng, clearance=1)

        # Assign realistic properties
        priority = rng.integers(1, 4)        # 1=low, 3=high
        base_value = rng.integers(10, 50)    # intrinsic reward
        release = rng.uniform(0, 5)          # random start window
        deadline = release + rng.uniform(30, 55)   # variable slack
        duration = rng.uniform(3.0, 7.0)     # different service times

        # Utility model
        urgency_term = w_d * (1.0 / (deadline - release + 1e-6))
        utility = (
            w_p * priority +
            w_v * base_value +
            urgency_term
        )

        tasks.append(
            Task(
                name=f"T{i+1}",
                location=(tx, ty),
                release_time=release,
                deadline=deadline,
                duration=duration,
                utility=utility,
            )
        )

    return grid_render, grid_plan, topo, robots, tasks, cell_to_region, region_to_cell


# =====================================================================
#   SCoBA
# =====================================================================

def run_scoba(robots, tasks):
    """
    SCoBA solver; returns plans and total utility.

    plans[robot.id] = [
       (task_name, start_t, finish_t, exp_contrib), ...
    ]
    """

    def tt(a, b):
        return travel_time_from_cells(a, b, speed=1.0)

    solver = SCoBASolver(robots, tasks, travel_time_func=tt)
    plans, total = solver.solve()

    print("\n=== SCoBA Allocation ===")
    print("Utility =", total)
    for r in robots:
        print(f"{r.id} :", plans[r.id])
    return plans


# =====================================================================
#   PM-CBS with conflict visualization hooks
# =====================================================================

def pmcbs_visual_marker(ax, conflict_regions, region_to_cell):
    """Draw PM-CBS conflict regions as red squares."""
    rects = []
    for rid in conflict_regions:
        x, y = region_to_cell[rid]
        r = Rectangle(
            (x, y),
            1,
            1,
            facecolor="none",
            edgecolor="red",
            linewidth=2.5,
            alpha=0.9,
            zorder=6,
        )
        ax.add_patch(r)
        rects.append(r)
    return rects


def run_pmcbs_current_goals(
    robots,
    robot_state,
    tasks,
    topo,
    cell_to_region,
    grid_plan,
    debug_cbs=False,
    ax=None,
    region_to_cell=None,
):
    """
    Computes conflict-free REGION paths for *all* robots, given
    their current task goals from robot_state.

    robot_state[robot.id] = {
        "plan":      list of Task objects (in SCoBA order),
        "task_idx":  current integer index (0..len(plan)),
    }

    If task_idx == len(plan): robot has no remaining tasks -> goal_region = current region.

    If debug_cbs and ax are provided, draws conflict region(s).
    """

    agent_ids = [r.id for r in robots]
    start_regions = {}
    goal_regions = {}

    h, w = cell_to_region.shape

    # compute start & goal
    for r in robots:
        rid = r.id
        state = robot_state[rid]
        plan = state["plan"]
        idx = state["task_idx"]

        # start region from pose
        x, y, _ = r.get_true_pose()
        sx = int(np.clip(int(np.floor(x)), 0, w - 1))
        sy = int(np.clip(int(np.floor(y)), 0, h - 1))
        sr = cell_to_region[sy, sx]

        if sr < 0:
            # fallback: nearest free region
            free = np.argwhere(cell_to_region >= 0)
            d = np.hypot(free[:, 1] - sx, free[:, 0] - sy)
            best = free[np.argmin(d)]
            sr = cell_to_region[best[0], best[1]]

        start_regions[rid] = sr

        # goal region
        if idx < len(plan):
            tx, ty = plan[idx].location
            gr = cell_to_region[ty, tx]
        else:
            gr = sr
        goal_regions[rid] = gr

    print("\n[PM-CBS] Replanning...")
    pm = PMCBS(
        topo,
        rspeed=1.0,
        I_margin=1.2,
        debug=debug_cbs,
        debug_grid=grid_plan,
    )

    # hook: capture conflict info (if debug_cbs is True)
    conflict_regions = None
    if debug_cbs and ax is not None and region_to_cell is not None:
        old_conflict_func = pm.detect_first_conflict

        def wrapped_conflict(*args, **kwargs):
            """
            Robust wrapper for PMCBS.detect_first_conflict.
            Accepts any argument signature.
            Captures region_id for visualization.
            """
            nonlocal conflict_regions
            out = old_conflict_func(*args, **kwargs)
            if out is not None:
                # out is a RegionConflict object
                try:
                    conflict_regions = [out.region_id]
                except Exception:
                    conflict_regions = None
            return out

        pm.detect_first_conflict = wrapped_conflict

    # Solve (with robust fallback)
    try:
        sol = pm.solve(agent_ids, start_regions, goal_regions)
    except Exception as e:
        print("[PM-CBS] ERROR:", e)
        print("[PM-CBS] Using independent region A* fallback for all agents.")
        sol = pm.fallback_independent_paths(agent_ids, start_regions, goal_regions)

    # draw conflict regions (if any)
    if ax is not None:
        # clear previous conflict patches if present
        old_patches = getattr(ax, "_pmcbs_conflict_patches", [])
        for p in old_patches:
            try:
                p.remove()
            except Exception:
                pass

        if conflict_regions and region_to_cell is not None:
            new_patches = pmcbs_visual_marker(ax, conflict_regions, region_to_cell)
        else:
            new_patches = []

        ax._pmcbs_conflict_patches = new_patches

    return sol


# =====================================================================
#   REGION → WAYPOINTS
# =====================================================================

def convert_region_paths_to_waypoints(paths, topo, grid_plan, region_to_cell):
    """
    paths[aid] = (region_path, times)

    For each consecutive pair of regions in region_path:
       - take their representative cells via region_to_cell[r]
       - run A* on the planning grid between those cells
       - stitch into (x+0.5, y+0.5) waypoints
    """
    per_agent = {}

    for aid, (rp, _) in paths.items():
        if len(rp) <= 1:
            per_agent[aid] = []
            continue

        full_cells = []

        for i in range(len(rp) - 1):
            r1 = rp[i]
            r2 = rp[i + 1]
            cx1, cy1 = region_to_cell[r1]
            cx2, cy2 = region_to_cell[r2]

            seg = grid_astar(grid_plan, (cx1, cy1), (cx2, cy2))
            if not seg:
                print(f"WARNING: No grid A* between region {r1} and {r2}")
                continue

            if i == 0:
                full_cells.extend(seg)
            else:
                full_cells.extend(seg[1:])  # avoid duplicate

        per_agent[aid] = [(x + 0.5, y + 0.5) for (x, y) in full_cells]

    return per_agent


# =====================================================================
#   CONTINUOUS SIMULATION WITH VISUALIZATION
# =====================================================================

def continuous_simulation(
    grid_render,
    grid_plan,
    topo,
    robots,
    tasks,
    cell_to_region,
    region_to_cell,
    scoba_plans=None,
    policy="scoba_cbs",
    dt=0.15,
    max_steps=2500,
    task_reach_eps=0.4,
    visualize=True,
    debug_cbs=False,
):
    """
    ONE continuous loop:

        - Robots move every step (unicycle + EKF, with obstacle safety).
        - High-level behavior depends on 'policy':

            * "scoba_cbs":
                - SCoBA per-robot plans
                - event-driven PM-CBS replanning

            * "greedy_astar":
                - nearest-task greedy wrt travel distance (no CBS)
                - tasks globally assigned once (placeholder hooks)

            * "rl1":
                - simple Q-learning over task choice (placeholder hooks)

        Returns a dict of metrics (for eval / plotting).
    """

    # ----- Common bookkeeping -----
    task_lookup = {t.name: t for t in tasks}
    completed_tasks = set()
    total_distance = {r.id: 0.0 for r in robots}
    prev_pos = {r.id: r.get_true_pose()[:2] for r in robots}
    collisions = 0
    collision_thresh = 0.3

    # RL baseline placeholders (kept for future use)
    q_table = {t.name: 0.0 for t in tasks}   # Q(task_name)
    alpha = 0.3                               # learning rate
    beta = 0.5                                # time penalty weight
    epsilon = 0.2                             # exploration prob

    # ----- POLICY-SPECIFIC STATE -----
    robot_state = {}
    tasks_available = set(t.name for t in tasks)  # tasks not yet claimed by any robot

    if policy == "scoba_cbs":
        if scoba_plans is None:
            raise ValueError("scoba_plans is required for policy='scoba_cbs'")

        # robot_state[robot_id] = {"plan": [Task,...], "task_idx": int}
        for r in robots:
            alloc = scoba_plans[r.id]  # list of (task_name, start_t, finish_t, contrib)
            plan_objs = [task_lookup[name] for (name, _, _, _) in alloc]
            robot_state[r.id] = {
                "plan": plan_objs,
                "task_idx": 0,
                "serving": False,
                "service_time_left": 0.0,
                "service_started_flag": False,  # for jitter-proofing
            }


        # Initial CBS
        paths = run_pmcbs_current_goals(
            robots,
            robot_state,
            tasks,
            topo,
            cell_to_region,
            grid_plan,
            debug_cbs=debug_cbs and visualize,
            ax=None,
            region_to_cell=region_to_cell,
        )
        wps = convert_region_paths_to_waypoints(
            paths, topo, grid_plan, region_to_cell
        )
        for r in robots:
            r.set_path(wps.get(r.id, []))

    elif policy in ("greedy_astar", "rl1"):
        # robot_state[robot_id] = {"current_task": None, "task_start_step": 0}
        for r in robots:
            robot_state[r.id] = {
                "current_task": None,
                "task_start_step": 0,
                "serving": False,
                "service_time_left": 0.0,
            }

    else:
        raise ValueError(f"Unknown policy '{policy}'")

    # ----- Visualization setup -----
    if visualize:
        h, w = grid_render.shape
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(grid_render, cmap="gray_r", origin="upper")

        draw_tasks(ax, tasks)

        # region graph (light)
        for rid, neigh in topo.neighbors.items():
            x1, y1 = topo.pos[rid]
            for nb in neigh:
                x2, y2 = topo.pos[nb]
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color="cyan",
                    alpha=0.03,
                    linewidth=0.3,
                    zorder=1,
                )

        scatters = []          # true pose markers
        trails = []            # true trajectory lines
        ekf_trails = []        # EKF estimated trajectory lines (black dashed)
        labels = []            # robot ID labels
        lidar_lines = []       # per-robot list of ray lines
        heading_true_lines = []
        heading_est_lines = []

        xs_hist = [[] for _ in robots]
        ys_hist = [[] for _ in robots]
        xs_est_hist = [[] for _ in robots]
        ys_est_hist = [[] for _ in robots]

        # initialize per-robot visuals
        NUM_LIDAR_RAYS = 30  # slight reduction for speed

        for i, r in enumerate(robots):
            col = agent_color(i)
            x, y, th = r.get_true_pose()
            ex, ey, eth = r.get_est_pose()

            # true pose marker
            sc = ax.scatter(
                x,
                y,
                s=120,
                color=col,
                edgecolors="black",
                zorder=6,
            )
            scatters.append(sc)

            # true path line
            ln_true, = ax.plot(
                [x],
                [y],
                color=col,
                linewidth=2.0,
                alpha=0.8,
                zorder=4,
            )
            trails.append(ln_true)

            # EKF estimated path line (black dashed)
            ln_est, = ax.plot(
                [ex],
                [ey],
                linestyle="--",
                color="black",
                linewidth=1.5,
                alpha=0.9,
                zorder=5,   # between rays and robot markers
            )
            ekf_trails.append(ln_est)

            # Robot label (black text with white box)
            lbl = ax.text(
                x,
                y + 0.3,
                r.id,
                color="black",
                fontsize=10,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                zorder=8,
            )
            labels.append(lbl)

            # Lidar initialization
            robot_lines = []
            try:
                init_ranges = r.lidar_scan(grid_render, num_rays=NUM_LIDAR_RAYS)
            except AttributeError:
                init_ranges = [0.0] * NUM_LIDAR_RAYS

            angles = th + np.linspace(-np.pi, np.pi, NUM_LIDAR_RAYS)
            for k, rng in enumerate(init_ranges):
                lx = [x, x + rng * np.cos(angles[k])]
                ly = [y, y + rng * np.sin(angles[k])]
                ln_lidar, = ax.plot(
                    lx,
                    ly,
                    color=col,
                    alpha=0.25,
                    linewidth=1.0,
                    zorder=2,
                )
                robot_lines.append(ln_lidar)
            lidar_lines.append(robot_lines)

            # Heading arrows (true + EKF), as reusable lines
            arrow_scale = 1.2
            hx = x + arrow_scale * np.cos(th)
            hy = y + arrow_scale * np.sin(th)
            ht_line, = ax.plot(
                [x, hx],
                [y, hy],
                color="white",
                linewidth=1.2,
                zorder=7,
            )
            heading_true_lines.append(ht_line)

            ehx = ex + arrow_scale * np.cos(eth)
            ehy = ey + arrow_scale * np.sin(eth)
            he_line, = ax.plot(
                [ex, ehx],
                [ey, ehy],
                color="yellow",
                linewidth=1.0,
                zorder=7,
            )
            heading_est_lines.append(he_line)

            xs_hist[i].append(x)
            ys_hist[i].append(y)
            xs_est_hist[i].append(ex)
            ys_est_hist[i].append(ey)

        ax.set_title("Continuous Simulation - Policy: " + policy.upper(), fontsize=16)
        plt.tight_layout()
        plt.show(block=False)
    else:
        fig = None
        ax = None
        scatters = trails = ekf_trails = labels = lidar_lines = None
        heading_true_lines = heading_est_lines = None
        xs_hist = ys_hist = xs_est_hist = ys_est_hist = None

    # ----- Main loop -----
    need_replan = (policy == "scoba_cbs")
    step = 0

    while step < max_steps:
        step += 1

        # 1) Move robots one time-step
        for i, r in enumerate(robots):
            rid = r.id
            state = robot_state[rid]

            old_x, old_y = prev_pos[r.id]

            # If robot is frozen at a task, DO NOT move it
            if state.get("locked_at_task", False):
                r.x_true[:] = r.x_true
                r.ekf.x[:] = r.x_true
            else:
                r.step(dt, grid_map=grid_render)

            x, y, th = r.get_true_pose()
            ex, ey, eth = r.get_est_pose()

            # distance travelled
            total_distance[r.id] += np.hypot(x - old_x, y - old_y)
            prev_pos[r.id] = (x, y)

            # Visualization update
            if visualize:
                # True path
                xs_hist[i].append(x)
                ys_hist[i].append(y)
                scatters[i].set_offsets([x, y])
                trails[i].set_data(xs_hist[i], ys_hist[i])

                # EKF path (black dashed)
                xs_est_hist[i].append(ex)
                ys_est_hist[i].append(ey)
                ekf_trails[i].set_data(xs_est_hist[i], ys_est_hist[i])

                # Label follows true robot
                labels[i].set_position((x, y + 0.25))

                # Lidar rays (update every frame; NUM_LIDAR_RAYS consistent)
                NUM_LIDAR_RAYS = len(lidar_lines[i])
                try:
                    ranges = r.lidar_scan(grid_render, num_rays=NUM_LIDAR_RAYS)
                except AttributeError:
                    ranges = [0.0] * NUM_LIDAR_RAYS

                ray_angles = th + np.linspace(-np.pi, np.pi, NUM_LIDAR_RAYS)
                for k, rng in enumerate(ranges):
                    lx = [x, x + rng * np.cos(ray_angles[k])]
                    ly = [y, y + rng * np.sin(ray_angles[k])]
                    lidar_lines[i][k].set_data(lx, ly)

                # Heading arrows (true & EKF)
                arrow_scale = 0.4
                # True
                hx = x + arrow_scale * np.cos(th)
                hy = y + arrow_scale * np.sin(th)
                heading_true_lines[i].set_data([x, hx], [y, hy])

                # EKF
                ehx = ex + arrow_scale * np.cos(eth)
                ehy = ey + arrow_scale * np.sin(eth)
                heading_est_lines[i].set_data([ex, ehx], [ey, ehy])

        # ------------------------------------------------------------
        # SCoBA: Detect arrival and START service
        # ------------------------------------------------------------
        if policy == "scoba_cbs":
            for r in robots:
                rid   = r.id
                state = robot_state[rid]
                plan  = state["plan"]
                idx   = state["task_idx"]

                if idx >= len(plan):  # no more tasks
                    continue

                current_task = plan[idx]
                tx, ty = current_task.location
                gx, gy = tx + 0.5, ty + 0.5

                x, y, _ = r.get_true_pose()
                dist = np.hypot(x - gx, y - gy)

                # ----------------------------------------------------
                # START SERVICE: only once per task
                # ----------------------------------------------------
                if dist < task_reach_eps and not state["serving"] and not state["service_started_flag"]:
                    state["serving"] = True
                    state["service_started_flag"] = True       # <--- YES THIS
                    state["service_time_left"] = current_task.duration
                    state["locked_at_task"] = True
                    r.current_wp_idx = len(r.path)             # freeze movement

                    print(f"[SIM] {rid} started service for {current_task.name} "
                        f"(duration={current_task.duration:.1f}s)")




        # Robot-robot collision check
        for i in range(len(robots)):
            xi, yi, _ = robots[i].get_true_pose()
            for j in range(i + 1, len(robots)):
                xj, yj, _ = robots[j].get_true_pose()
                if np.hypot(xi - xj, yi - yj) < collision_thresh:
                    collisions += 1

        # 2) High-level task logic per policy
        if policy == "scoba_cbs":
            all_done = False
            # -----------------------------------------------------
            # 1) SERVICE LOGIC (SCoBA only) — JITTER-PROOF VERSION
            # -----------------------------------------------------
            for r in robots:
                rid = r.id
                state = robot_state[rid]

                # Initialize locking flag if missing
                if "locked_at_task" not in state:
                    state["locked_at_task"] = False

                # -------------------------------------------------
                # If robot is serving: freeze pose + countdown
                # -------------------------------------------------
                if state["serving"]:
                    state["service_time_left"] -= dt

                    # Freeze robot completely (ignore EKF/motion noise)
                    # We directly overwrite the pose:
                    r.x_true[:] = r.x_true  # no change
                    r.ekf.x[:] = r.x_true   # perfect estimate when frozen
                    r.current_wp_idx = len(r.path)  # disable path following
                    state["locked_at_task"] = True

                    if state["service_time_left"] > 0:
                        continue  # stay frozen until service is done

                    # SERVICE FINISHED
                    state["serving"] = False
                    state["locked_at_task"] = False
                    print(f"[SIM] {rid} completed service for {state['plan'][state['task_idx']].name}")

                    done_task = state["plan"][state["task_idx"]]
                    completed_tasks.add(done_task.name)
                    state["task_idx"] += 1
                    state["service_started_flag"] = False

                    # Only replan IF robot still has remaining tasks
                    if state["task_idx"] < len(state["plan"]):
                        need_replan = True
                    else:
                        need_replan = False


                    # DO NOT continue — execution falls through normally
                    all_done = True
                    for rr in robots:
                        sid = rr.id
                        st = robot_state[sid]
                        if st["task_idx"] < len(st["plan"]):
                            all_done = False
                            break


        elif policy in ("greedy_astar", "rl1"):
            # ==========================================================
            # TRUE GREEDY A* BASELINE + RL1 VARIANT
            # ----------------------------------------------------------
            # - NO task exclusivity
            # - NO coordination
            # - NO PM-CBS
            # - Each robot repeatedly picks the nearest unfinished task
            # - Multiple robots may pick SAME task
            # - Collisions expected (baseline weakness)
            # - Tasks only disappear when physically completed
            # ==========================================================

            all_done = True   # will flip to False if any unfinished task

            for r in robots:
                rid = r.id
                state = robot_state[rid]

                # If robot finished its path *and is not serving*, reset its target
                if r.has_finished_path() and not state["serving"]:
                    state["current_task"] = None

                # If robot has no current target → assign one
                if state["current_task"] is None:

                    # Get list of unfinished tasks
                    unfinished = [t for t in tasks
                                  if t.name not in completed_tasks]

                    if not unfinished:
                        continue   # nothing left for this robot

                    all_done = False

                    # Robot's current cell
                    x, y, _ = r.get_true_pose()
                    sx = int(np.floor(x))
                    sy = int(np.floor(y))

                    # -------------------------------
                    # GREEDY = pick nearest task
                    # -------------------------------
                    if policy == "greedy_astar":
                        best_t = None
                        best_cost = float("inf")

                        for t in unfinished:
                            tx, ty = t.location
                            cost = abs(tx - sx) + abs(ty - sy)
                            if cost < best_cost:
                                best_cost = cost
                                best_t = t

                        chosen = best_t.name

                    # -------------------------------
                    # RL1 = ε-greedy soft selection
                    # -------------------------------
                    else:  # rl1
                        remaining_names = [t.name for t in unfinished]

                        if np.random.rand() < epsilon:  # explore
                            chosen = np.random.choice(remaining_names)
                        else:  # exploit
                            best_q = -float('inf')
                            chosen = None
                            for tname in remaining_names:
                                if q_table[tname] > best_q:
                                    best_q = q_table[tname]
                                    chosen = tname

                    # Store the chosen task
                    state["current_task"] = chosen
                    state["task_start_step"] = step

                    # Plan A* path (NO reservation)
                    t = task_lookup[chosen]
                    tx, ty = t.location
                    seg = grid_astar(grid_plan, (sx, sy), (tx, ty))

                    if seg:
                        wps = [(cx + 0.5, cy + 0.5) for (cx, cy) in seg]
                        r.set_path(wps)
                    else:
                        # No path available → idle (baseline remains dumb)
                        r.set_path([])

            # ======================================================
            #   2) CHECK COMPLETION OF TASKS
            # ======================================================
            for r in robots:
                rid = r.id
                state = robot_state[rid]
                cur_name = state["current_task"]

                # If currently serving, tick down the service timer
                if state["serving"]:
                    state["service_time_left"] -= dt
                    if state["service_time_left"] < 0:
                        state["service_time_left"] = 0.0


                if cur_name is None:
                    continue

                t = task_lookup[cur_name]
                tx, ty = t.location
                gx, gy = tx + 0.5, ty + 0.5

                x, y, _ = r.get_true_pose()
                dist = np.hypot(x - gx, y - gy)

                if dist < task_reach_eps:
                    # Start service if not already serving
                    if not state["serving"]:
                        state["serving"] = True
                        state["service_time_left"] = t.duration
                        # freeze path so it doesn't wander
                        r.set_path([])
                        print(f"[SIM] {rid} START service {cur_name} "
                            f"(duration={t.duration:.1f}s)")
                        continue  # stay here until service finishes


                    # Finish service
                    if state["serving"] and state["service_time_left"] <= 0:
                        print(f"[SIM] {rid} FINISH service {cur_name} ({policy})")
                        state["serving"] = False
                        completed_tasks.add(cur_name)

                        if policy == "rl1":
                            elapsed = (step - state["task_start_step"]) * dt
                            reward = t.utility - beta * elapsed
                            old_q = q_table[cur_name]
                            q_table[cur_name] = (1 - alpha) * old_q + alpha * reward

                        state["current_task"] = None
                        r.set_path([])


            # ======================================================
            #   3) END CONDITION
            # ======================================================
            if len(completed_tasks) == len(tasks):
                all_done = True
            else:
                all_done = False

        # 3) PM-CBS replanning if needed
        if policy == "scoba_cbs" and need_replan:
            paths = run_pmcbs_current_goals(
                robots,
                robot_state,
                tasks,
                topo,
                cell_to_region,
                grid_plan,
                debug_cbs=debug_cbs and visualize,
                ax=ax if visualize else None,
                region_to_cell=region_to_cell,
            )
            wps = convert_region_paths_to_waypoints(
                paths, topo, grid_plan, region_to_cell
            )
            for r in robots:
                r.set_path(wps.get(r.id, []))
            need_replan = False

        if all_done:
            print("[SIM] All tasks completed or no more tasks reachable.")
            break

        if visualize:
            plt.pause(0.03)

    if visualize and fig is not None:
        plt.close(fig)

    # ----- Metrics -----
    total_tasks = len(tasks)
    completed_list = list(completed_tasks)
    total_utility = sum(task_lookup[name].utility for name in completed_list)

    metrics = {
        "policy": policy,
        "steps": step,
        "total_distance": sum(total_distance.values()),
        "collisions": collisions,
        "tasks_completed": len(completed_list),
        "total_tasks": total_tasks,
        "fraction_tasks_completed": (
            len(completed_list) / total_tasks if total_tasks > 0 else 0.0
        ),
        "total_utility": total_utility,
    }
    return metrics


# =====================================================================
#   UNIFIED EPISODE RUNNER (with headless option)
# =====================================================================

def run_episode(
    policy_type,
    seed=None,
    visualize=True,
    headless=None,
    num_agents=3,
    num_tasks=6,
    dt=0.15,
    max_steps=500,
    task_reach_eps=0.4,
    debug_cbs=False,
):
    """
    Wrapper around continuous_simulation used by eval_policies.py.

    If headless is provided → override visualize.
    This keeps backward compatibility for both evaluators and main().
    """
    if headless is not None:
        visualize = not headless

    grid_render, grid_plan, topo, robots, tasks, cell_to_region, region_to_cell = \
        build_scenario(num_agents=num_agents, num_tasks=num_tasks, seed=seed)

    scoba_plans = None
    if policy_type == "scoba_cbs":
        scoba_plans = run_scoba(robots, tasks)

    metrics = continuous_simulation(
        grid_render,
        grid_plan,
        topo,
        robots,
        tasks,
        cell_to_region,
        region_to_cell,
        scoba_plans=scoba_plans,
        policy=policy_type,
        dt=dt,
        max_steps=max_steps,
        task_reach_eps=task_reach_eps,
        visualize=visualize,
        debug_cbs=debug_cbs,
    )
    return metrics


# =====================================================================
#   MAIN (runs one visual episode with OUR policy)
# =====================================================================

def main():
    metrics = run_episode(
        policy_type="scoba_cbs",  # "scoba_cbs" or "greedy_astar" or "rl1"
        seed=random.randint(0, 10000),
        num_agents=5,
        num_tasks=7,
        visualize=True,     # SHOW animation
        headless=None,      # for convenience
        dt=0.3,             # slightly faster dynamics
        max_steps=500,
        task_reach_eps=0.4,
        debug_cbs=False,    # enable conflict-region visualization in PM-CBS
    )

    print("\n=== RUN COMPLETE ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
