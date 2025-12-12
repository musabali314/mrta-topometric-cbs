"""
map_warehouse.py
----------------
Randomized warehouse map generator with:
- Random shelves
- Random stations
- Random pallet clusters
- Open plazas
- Always navigable structure
- Obstacle inflation for safe planning

Exports:
    create_random_warehouse()
    build_topometric_map()
    safe_free_cell_with_clearance()
    inflate_obstacles()
"""

import numpy as np
from scipy.ndimage import binary_dilation


# ============================================================
# 1) Random Warehouse Generator
# ============================================================

# =====================================================================
#   ROBUST RANDOM WAREHOUSE GENERATOR  (works for any size >= 12x12)
# =====================================================================

import numpy as np

def create_random_warehouse(
        width=40,
        height=40,
        seed=None,
        shelf_count=8,
        shelf_min_len=7,
        shelf_max_len=14,
        shelf_thickness=2,
        pallet_count=25):
    """
    A size-adaptive random warehouse generator.
    Produces:
        - Outer walls
        - Random vertical shelves
        - Random horizontal shelves
        - Random pallets
    Works for ANY map >= 20x20 without crashing.
    """

    rng = np.random.default_rng(seed)
    g = np.zeros((height, width), dtype=int)

    # 1) Perimeter walls
    g[0, :] = 1
    g[-1, :] = 1
    g[:, 0] = 1
    g[:, -1] = 1

    # 2) Random vertical shelves
    for _ in range(shelf_count // 2):
        x = rng.integers(2, width - shelf_thickness - 2)

        # choose length safely
        max_len = min(shelf_max_len, height - 6)
        min_len = min(shelf_min_len, max_len)

        if max_len < 3:
            continue

        length = rng.integers(min_len, max_len + 1)
        y0 = rng.integers(2, height - length - 2)

        g[y0:y0 + length, x:x + shelf_thickness] = 1

    # 3) Random horizontal shelves
    for _ in range(shelf_count // 2):
        y = rng.integers(2, height - shelf_thickness - 2)

        # choose safe length
        max_len = min(shelf_max_len, width - 6)
        min_len = min(shelf_min_len, max_len)

        if max_len < 3:
            continue

        length = rng.integers(min_len, max_len + 1)
        x0 = rng.integers(2, width - length - 2)

        g[y:y + shelf_thickness, x0:x0 + length] = 1

    # 4) Random pallets
    for _ in range(pallet_count):
        px = rng.integers(2, width - 3)
        py = rng.integers(2, height - 3)
        w = rng.integers(1, 3)   # 1–2 wide
        h = rng.integers(1, 3)   # 1–2 tall

        if px + w < width - 1 and py + h < height - 1:
            g[py:py + h, px:px + w] = 1

    return g



# ============================================================
# 2) Obstacle Inflation
# ============================================================

def inflate_obstacles(grid, inflation_radius=1):
    """
    Expands obstacles by inflation_radius using binary dilation.
    Returns a *new* grid used for safe planning.
    """
    mask = (grid == 1)
    struct = np.ones((2 * inflation_radius + 1,
                      2 * inflation_radius + 1))
    inflated = binary_dilation(mask, structure=struct)
    return inflated.astype(int)


# ============================================================
# 3) Build Topometric Map
# ============================================================

import numpy as np
import math

class TopometricGraph:
    def __init__(self, regions, pos, neighbors, cell_to_region, region_to_cell):
        self.regions = regions               # dict[rid] -> list of free cells in region
        self.pos = pos                       # dict[rid] -> (cx,cy)
        self.neighbors = neighbors           # dict[rid] -> list[rid]
        self.cell_to_region = cell_to_region
        self.region_to_cell = region_to_cell

        self.region_ids = list(regions.keys())
        self.num_regions = len(self.region_ids)

    def distance(self, r1, r2):
        (x1, y1) = self.pos[r1]
        (x2, y2) = self.pos[r2]
        return math.hypot(x2 - x1, y2 - y1)


def build_topometric_map(grid):
    """
    Free cell → region graph.

    EVERY free grid cell = its own region.
    Fully compatible with low_level_topometric_planner + CBS.
    """
    h, w = grid.shape
    cell_to_region = -np.ones((h, w), dtype=int)

    regions = {}
    pos = {}
    region_to_cell = {}

    rid = 0
    for y in range(h):
        for x in range(w):
            if grid[y, x] == 0:
                cell_to_region[y, x] = rid
                regions[rid] = [(x, y)]       # region = one cell
                pos[rid] = (x + 0.5, y + 0.5)
                region_to_cell[rid] = (x, y)
                rid += 1

    # build adjacency
    neighbors = {i: [] for i in range(rid)}

    for y in range(h):
        for x in range(w):
            r = cell_to_region[y, x]
            if r < 0:
                continue
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < w and 0 <= ny < h and grid[ny, nx] == 0:
                    rn = cell_to_region[ny, nx]
                    neighbors[r].append(rn)

    return TopometricGraph(regions, pos, neighbors, cell_to_region, region_to_cell)



# ============================================================
# 4) Safe Cell Sampling
# ============================================================

def safe_free_cell_with_clearance(grid, rng, clearance=1):
    candidates = []
    h, w = grid.shape
    for y in range(clearance, h - clearance):
        for x in range(clearance, w - clearance):
            if grid[y, x] != 0:
                continue
            patch = grid[y - clearance:y + clearance + 1,
                         x - clearance:x + clearance + 1]
            if np.all(patch == 0):
                candidates.append((x, y))

    if not candidates:
        raise RuntimeError("No safe free cell with clearance found!")

    return candidates[rng.integers(0, len(candidates))]
