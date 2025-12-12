"""
cbs_topometric.py
-----------------
PM-CBS with optional search visualization.

Use:
    pm = PMCBS(topo_graph, debug=True)
"""

import heapq
import time
from copy import deepcopy

import numpy as np

from low_level_topometric_planner import LowLevelTopometricPlanner
from cbs_debugger import CBSDebugger   # <--- NEW


# ============================================================
# Conflict structures
# ============================================================

class RegionConflict:
    def __init__(self, ai, aj, region_id, ti0, ti1, tj0, tj1):
        self.ai = ai
        self.aj = aj
        self.region_id = region_id
        self.ti0 = ti0
        self.ti1 = ti1
        self.tj0 = tj0
        self.tj1 = tj1

    def conflict_time(self):
        return max(self.ti0, self.tj0)


class EdgeConflict:
    def __init__(self, ai, aj, r1, r2, ti0, ti1, tj0, tj1):
        self.ai = ai
        self.aj = aj
        self.r1 = r1
        self.r2 = r2
        self.ti0 = ti0
        self.ti1 = ti1
        self.tj0 = tj0
        self.tj1 = tj1

    def conflict_time(self):
        return max(self.ti0, self.tj0)


# ============================================================
# PM-CBS class
# ============================================================

class PMCBS:
    def __init__(
        self,
        topo_graph,
        rspeed=1.0,
        I_margin=1.2,
        max_nodes=8000,
        max_time=0.5,
        debug=False,
        debug_grid=None,
    ):
        """
        topo_graph : TopometricGraph
        rspeed     : nominal region-level speed
        I_margin   : inflation factor
        max_nodes  : maximum CT nodes before fallback
        max_time   : maximum wall-clock time (seconds) before fallback
        debug      : enable CBS debugger visualization
        """
        self.topo = topo_graph
        self.rspeed = rspeed
        self.I_margin = I_margin
        self.max_nodes = max_nodes
        self.max_time = max_time

        self.low = LowLevelTopometricPlanner(
            topo_graph,
            rspeed=rspeed,
            I_margin=I_margin,
        )

        # debugger
        self.debug = debug
        if debug:
            assert debug_grid is not None, "debug_grid must be passed when debug=True"
            self.dbg = CBSDebugger(topo_graph, debug_grid)
        else:
            self.dbg = None

    # -----------------------------------------------------------
    # Simple region-level fallback A*
    # -----------------------------------------------------------

    def simple_region_astar(self, sr, gr):
        """
        VERY simple region-level A* used in fallback_independent_paths.
        Operates on self.topo.neighbors and self.topo.distance.
        """
        if sr == gr:
            return [sr]

        open_heap = []
        heapq.heappush(open_heap, (0.0, sr))
        came_from = {sr: None}
        g_cost = {sr: 0.0}

        def h(r):
            return self.topo.distance(r, gr)

        while open_heap:
            _, r = heapq.heappop(open_heap)
            if r == gr:
                # reconstruct
                path = []
                cur = r
                while cur is not None:
                    path.append(cur)
                    cur = came_from[cur]
                return path[::-1]

            for nb in self.topo.neighbors.get(r, []):
                step_cost = self.topo.distance(r, nb)
                new_g = g_cost[r] + step_cost
                if nb not in g_cost or new_g < g_cost[nb]:
                    g_cost[nb] = new_g
                    came_from[nb] = r
                    f = new_g + h(nb)
                    heapq.heappush(open_heap, (f, nb))

        # If we totally fail, at least return [sr, gr] so the sim doesn't crash.
        return [sr, gr] if sr != gr else [sr]

    # -----------------------------------------------------------
    # Interval helpers
    # -----------------------------------------------------------

    @staticmethod
    def region_intervals(path, times):
        out = []
        for i in range(len(path)):
            r = path[i]
            t0 = times[i]
            t1 = times[i + 1] if i < len(path) - 1 else times[i]
            out.append((r, t0, t1))
        return out

    @staticmethod
    def edge_intervals(path, times):
        out = []
        for i in range(len(path) - 1):
            r1 = path[i]
            r2 = path[i + 1]
            t1 = times[i]
            t2 = times[i + 1]
            out.append((r1, r2, t1, t2))
        return out

    # -----------------------------------------------------------
    # Conflict detection + fallback
    # -----------------------------------------------------------

    def fallback_independent_paths(self, agent_ids, start_regions, goal_regions):
        """
        When CBS fails / times out, we just plan independent region paths
        for each agent using simple_region_astar (no temporal reasoning).
        """
        print("[PM-CBS] Using independent region A* fallback for all agents.")
        paths = {}
        for aid in agent_ids:
            sr = start_regions[aid]
            gr = goal_regions[aid]
            rp = self.simple_region_astar(sr, gr)
            times = np.arange(len(rp)) * self.rspeed
            paths[aid] = (rp, times)
        return paths

    def detect_first_conflict(self, paths):
        reg_int = {}
        edge_int = {}
        for aid, (rp, ts) in paths.items():
            reg_int[aid] = self.region_intervals(rp, ts)
            edge_int[aid] = self.edge_intervals(rp, ts)

        best = None
        best_t = float("inf")
        agents = list(paths.keys())

        # region conflicts
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                ai = agents[i]
                aj = agents[j]
                for (ri, ti0, ti1) in reg_int[ai]:
                    for (rj, tj0, tj1) in reg_int[aj]:
                        if ri != rj:
                            continue
                        overlap = min(ti1, tj1) - max(ti0, tj0)
                        if overlap > 0:
                            t_conf = max(ti0, tj0)
                            if t_conf < best_t:
                                best_t = t_conf
                                best = RegionConflict(ai, aj, ri, ti0, ti1, tj0, tj1)

        # edge conflicts
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                ai = agents[i]
                aj = agents[j]
                for (r1i, r2i, ti0, ti1) in edge_int[ai]:
                    for (r1j, r2j, tj0, tj1) in edge_int[aj]:
                        if r1i == r2j and r2i == r1j:
                            overlap = min(ti1, tj1) - max(ti0, tj0)
                            if overlap > 0:
                                t_conf = max(ti0, tj0)
                                if t_conf < best_t:
                                    best = EdgeConflict(
                                        ai, aj, r1i, r2i, ti0, ti1, tj0, tj1
                                    )
                                    best_t = t_conf
        return best

    # -----------------------------------------------------------
    # Cost
    # -----------------------------------------------------------

    @staticmethod
    def cost(paths):
        total = 0
        for _, (_, ts) in paths.items():
            if ts:
                total += ts[-1]
        return total

    # -----------------------------------------------------------
    # Solve
    # -----------------------------------------------------------

    def solve(self, agent_ids, start_regions, goal_regions):
        """
        Standard CBS on region-time space with:

          - Node limit (self.max_nodes)
          - Wall-clock time limit (self.max_time)
          - Independent-paths fallback on failure
        """
        start_wall = time.time()

        # root constraints
        cons = {aid: {} for aid in agent_ids}

        # initial paths
        paths = {}
        for aid in agent_ids:
            sr = start_regions[aid]
            gr = goal_regions[aid]
            rp, ts = self.low.plan(sr, gr, cons[aid])
            if rp is None:
                print(f"[PM-CBS] Root low-level failure for agent {aid} → fallback")
                return self.fallback_independent_paths(agent_ids, start_regions, goal_regions)
            paths[aid] = (rp, ts)

        root_cost = self.cost(paths)
        CT = [(root_cost, 0, cons, paths)]
        node_id = 1
        expansions = 0

        while CT:
            # Time limit check
            if time.time() - start_wall > self.max_time:
                print("[PM-CBS] TIME LIMIT exceeded → fallback")
                return self.fallback_independent_paths(agent_ids, start_regions, goal_regions)

            if expansions >= self.max_nodes:
                print("[PM-CBS] NODE LIMIT reached → fallback")
                return self.fallback_independent_paths(agent_ids, start_regions, goal_regions)

            cost, nid, cons, paths = heapq.heappop(CT)
            expansions += 1

            # ⚡ Debug: visualize expansion of all path heads
            if self.debug:
                for aid, (rp, _) in paths.items():
                    if rp:
                        self.dbg.draw_expansion(rp[0])

            conflict = self.detect_first_conflict(paths)
            if conflict is None:
                # success: draw final paths
                if self.debug:
                    for aid, (rp, _) in paths.items():
                        self.dbg.draw_final_path(rp)
                return paths

            # ⚡ Debug: draw conflict
            if self.debug:
                if isinstance(conflict, RegionConflict):
                    self.dbg.draw_conflict(conflict.region_id)
                else:
                    self.dbg.draw_conflict(conflict.r1)

            # create children
            if isinstance(conflict, RegionConflict):
                ai, aj = conflict.ai, conflict.aj
                r = conflict.region_id

                c1 = deepcopy(cons)
                c1[ai].setdefault(r, []).append((conflict.tj0, conflict.tj1))

                c2 = deepcopy(cons)
                c2[aj].setdefault(r, []).append((conflict.ti0, conflict.ti1))

                branches = [(ai, c1), (aj, c2)]

            else:
                ai, aj = conflict.ai, conflict.aj
                r1 = conflict.r1
                r2 = conflict.r2
                pad = 0.01

                c1 = deepcopy(cons)
                c1[ai].setdefault(r2, []).append(
                    (conflict.tj0 - pad, conflict.tj1 + pad)
                )
                c2 = deepcopy(cons)
                c2[aj].setdefault(r1, []).append(
                    (conflict.ti0 - pad, conflict.ti1 + pad)
                )
                branches = [(ai, c1), (aj, c2)]

            # replan only affected
            for agent, new_cons in branches:
                new_paths = dict(paths)

                sr = start_regions[agent]
                gr = goal_regions[agent]
                rp, ts = self.low.plan(sr, gr, new_cons[agent])
                if rp is None:
                    continue

                new_paths[agent] = (rp, ts)
                heapq.heappush(
                    CT,
                    (
                        self.cost(new_paths),
                        node_id,
                        new_cons,
                        new_paths,
                    ),
                )
                node_id += 1

        print("[PM-CBS] WARNING: search exhausted – falling back to independent A*")
        return self.fallback_independent_paths(agent_ids, start_regions, goal_regions)
