"""
low_level_topometric_planner.py
-------------------------------

Single-agent low-level planner over a topometric graph with
time-slot constraints, inspired by Section III-C of the PM-CBS paper.

This module is now designed to work with the topometric map built by:
    map_warehouse.build_topometric_map(grid)

It provides:
    - TopometricGraph     : thin wrapper around the topo dict
    - compute_time_slots  : forbidden-interval -> allowed-slots
    - LowLevelTopometricPlanner : A* in (region, time) space
    - travel_time_from_cells    : simple grid-cell travel-time heuristic
"""

import heapq
import math


# ============================================================
#           TOPOGRAPH / REGION GRAPH WRAPPER
# ============================================================

class TopometricGraph:
    """
    Wrapper over the 'topo' dict produced by map_warehouse.build_topometric_map.

    Expected keys in topo:
        topo["regions"]        : dict[region_id] -> list[(x, y), ...]
        topo["region_centers"] : dict[region_id] -> (cx, cy)  (float)
        topo["neighbors"]      : dict[region_id] -> set/list[neighbor_region_id]

    This class exposes:
        - num_regions
        - region_ids
        - pos[region_id]       : (cx, cy)
        - neighbors[region_id] : list of region_ids
        - distance(r1, r2)     : Euclidean distance between region centers
    """

    def __init__(self, topo):
        self.regions = topo["regions"]                 # dict[rid] -> list[(x,y)]
        self.pos = topo["region_centers"]              # dict[rid] -> (cx, cy)
        raw_neighbors = topo["neighbors"]              # dict[rid] -> set or list

        # Normalize neighbors to plain lists
        self.neighbors = {
            rid: list(nbrs) for rid, nbrs in raw_neighbors.items()
        }

        # Region IDs (usually 0..N-1)
        self.region_ids = list(self.regions.keys())
        self.num_regions = len(self.region_ids)

    def distance(self, r1, r2):
        """Euclidean distance between representative centers of two regions."""
        x1, y1 = self.pos[r1]
        x2, y2 = self.pos[r2]
        return math.hypot(x2 - x1, y2 - y1)


# ============================================================
#          TIME-SLOT / CONSTRAINT UTILITIES
# ============================================================

def compute_time_slots(constraints, t_max=1e6):
    """
    Given a list of forbidden intervals [ (start, end), ... ]
    returns a list of allowed intervals (time slots).

    This implements SA_k = (0, ∞) \ C_Ak in a truncated way.
    """
    if not constraints:
        return [(0.0, t_max)]

    cons = sorted(constraints, key=lambda c: c[0])
    slots = []
    current_start = 0.0

    for s, e in cons:
        if s > current_start:
            slots.append((current_start, s))
        current_start = max(current_start, e)

    if current_start < t_max:
        slots.append((current_start, t_max))

    return slots


# ============================================================
#              LOW-LEVEL TOPOGRAPH PLANNER
# ============================================================

class LowLevelTopometricPlanner:
    """
    Single-agent planner over a TopometricGraph with region time constraints.

    We do A* in (region, time) space:
       - cost = arrival_time
       - we respect per-region allowed time slots (no entry during forbidden intervals)
    """

    def __init__(self, topo_graph, rspeed=1.0, I_margin=1.2, t_max=1e6):
        """
        topo_graph : TopometricGraph
        rspeed     : nominal robot speed (units / second)
        I_margin   : time inflation factor
        t_max      : max planning horizon
        """
        self.g = topo_graph
        self.rspeed = rspeed
        self.I_margin = I_margin
        self.t_max = t_max

    def plan(self, start_region, goal_region, region_constraints):
        """
        Plan a path from start_region to goal_region subject to
        region time constraints.

        region_constraints:
            dict[region_id] -> list[(t_start, t_end)] of FORBIDDEN intervals.

        Returns:
            path_regions: [r0, r1, ..., rK]
            times      : [t_arrive_r0, t_arrive_r1, ..., t_arrive_rK]

        or (None, None) if no path exists.
        """

        # Precompute allowed time slots for the regions
        region_slots = {
            r: compute_time_slots(region_constraints.get(r, []), self.t_max)
            for r in self.g.region_ids
        }

        # Priority queue: (f, arrival_time, region, parent_key)
        open_heap = []
        parents = {}  # key -> (region, t_arrive, parent_key)

        def key_of(region, t_arrive):
            # coarse key to avoid exploding state space
            return (region, round(t_arrive, 2))

        # Initialize at start_region at t = 0 within a valid slot
        if start_region not in region_slots:
            return None, None

        for (s, e) in region_slots[start_region]:
            if e <= 0:
                continue
            t_start = max(0.0, s)
            if t_start < e:
                root_key = key_of(start_region, t_start)
                parents[root_key] = (start_region, t_start, None)

                h = self.g.distance(start_region, goal_region) / self.rspeed * self.I_margin
                f0 = t_start + h
                heapq.heappush(open_heap, (f0, t_start, start_region, root_key))
                break
        else:
            # No feasible initial slot
            return None, None

        visited = set()

        while open_heap:
            f, t_curr, r_curr, node_key = heapq.heappop(open_heap)

            if node_key in visited:
                continue
            visited.add(node_key)

            if r_curr == goal_region:
                # Reconstruct path
                path_regions = []
                times = []
                k = node_key
                while k is not None:
                    r, t_arr, p_k = parents[k]
                    path_regions.append(r)
                    times.append(t_arr)
                    k = p_k
                path_regions.reverse()
                times.reverse()
                return path_regions, times

            # Expand neighbors
            for r_next in self.g.neighbors.get(r_curr, []):
                d = self.g.distance(r_curr, r_next)
                travel_time = (d / self.rspeed) * self.I_margin

                for (slot_start, slot_end) in region_slots[r_next]:
                    earliest_enter = max(t_curr, slot_start)
                    arrive_time = earliest_enter + travel_time

                    if arrive_time > slot_end:
                        # Can't traverse r_next within this slot
                        continue

                    succ_key = key_of(r_next, arrive_time)
                    if succ_key in visited:
                        continue

                    # Record better parent (lower arrival time)
                    if succ_key not in parents or parents[succ_key][1] > arrive_time:
                        parents[succ_key] = (r_next, arrive_time, node_key)
                        h = self.g.distance(r_next, goal_region) / self.rspeed * self.I_margin
                        f_succ = arrive_time + h
                        heapq.heappush(open_heap, (f_succ, arrive_time, r_next, succ_key))

        # Exhausted search
        return None, None


# ============================================================
#     TRAVEL-TIME HELPER FOR SCOBA / HIGH-LEVEL HEURISTICS
# ============================================================

def travel_time_from_cells(locA, locB, speed=1.0):
    """
    Simple travel-time function based on Manhattan distance in grid cells.

    locA, locB : (x, y) in grid indices.

    This does NOT use the topometric graph directly; it is
    only meant as an approximate heuristic for SCoBA utilities.
    """
    (x1, y1) = locA
    (x2, y2) = locB
    dist = abs(x1 - x2) + abs(y1 - y2)
    return dist / max(1e-6, speed)
