"""Food sensing: range, density, and trail/cluster targeting.

The game camera is often ~400 units at spawn while the observation matrix
covers 1000 and sectors cover 2000. Food must be collected out to
FOOD_SENSE_RANGE. Adjacent occupied cells form a string; targeting aims at
the nearest pellet on the richest string so the snake follows trails instead
of cutting toward a centroid in empty space.
"""

import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

FOOD_SENSE_RANGE = 2000.0
FOOD_CLUSTER_CELL = 120.0
# Hold a cluster while its centroid stays within two cells; stops shaping
# from flipping between two nearby piles.
FOOD_LOCK_RADIUS = 240.0
# Per-step food/prey cap. 800 covers a busy 2000-unit disc; lower it via
# SLBOT_MAX_FOODS or --max-foods if payload/paint cost shows up in St/s.
MAX_FOODS_DEFAULT = 800
MAX_FOODS_MIN = 50
MAX_FOODS_MAX = 4000
FOOD_LIST_PRUNE_MULT = 3


def configured_max_foods(default: int = MAX_FOODS_DEFAULT) -> int:
    """Runtime food cap: SLBOT_MAX_FOODS env, else default. Clamped."""
    raw = os.environ.get("SLBOT_MAX_FOODS", "").strip()
    if not raw:
        return int(default)
    try:
        return max(MAX_FOODS_MIN, min(MAX_FOODS_MAX, int(raw)))
    except ValueError:
        return int(default)


MAX_FOODS = configured_max_foods()
FOOD_CHANNEL_LOG_CAP = 20.0
FOOD_SECTOR_LOG_CAP = 12.0
FOOD_KEEP_DIST_BIAS = 80.0
# World-units of paint radius per food.sz, converted by matrix scale.
# Crumbs stay ~1 pixel; a sz=12 pellet is a several-pixel blob at 160px.
FOOD_DRAW_RADIUS_PER_SZ = 4.0
FOOD_DRAW_RADIUS_MIN_PX = 0.85
PREY_DEFAULT_SIZE = 10.0
# Prey can slide this far in one env step; static pellets stay put.
EAT_MATCH_RADIUS = 45.0
CLUSTER_EAT_CRUMB = 1.0
CLUSTER_EAT_MASS_CAP = 40.0
BOOST_CLUSTER_RANGE = 400.0
BOOST_CLUSTER_MIN_MASS = 6.0
_NEIGHBOR_8 = tuple(
    (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy
)

FoodItem = Sequence[float]


def food_size(item: FoodItem) -> float:
    if item is None or len(item) < 2:
        return 0.0
    if len(item) > 2:
        try:
            sz = float(item[2])
        except (TypeError, ValueError):
            sz = 1.0
        return sz if sz > 0.0 else 1.0
    return 1.0


def squash_mass(mass: float, cap: float) -> float:
    """Map raw mass to (0, 1] with log compression so piles outrank crumbs."""
    if mass <= 0.0 or cap <= 0.0:
        return 0.0
    return min(1.0, math.log1p(mass) / math.log1p(cap))


def food_draw_radius_px(sz, scale):
    """Matrix-pixel radius so large pellets occupy more cells than crumbs.

    Accepts scalars or numpy arrays for `sz` (and broadcast-compatible
    `scale`) so the scalar and vectorized draw paths share one formula.
    """
    return np.maximum(FOOD_DRAW_RADIUS_MIN_PX, sz * FOOD_DRAW_RADIUS_PER_SZ * scale)


def cluster_eat_bonus(
    mass: float,
    scale: float,
    crumb: float = CLUSTER_EAT_CRUMB,
    cap: float = CLUSTER_EAT_MASS_CAP,
) -> float:
    """Extra reward for eating more than a lone crumb. Log-compressed and capped."""
    if scale <= 0.0 or mass <= crumb:
        return 0.0
    surplus = min(mass - crumb, cap)
    return scale * math.log1p(surplus)


def boost_cluster_bonus(
    dist: Optional[float],
    mass: float,
    closing: bool,
    scale: float,
    range_: float = BOOST_CLUSTER_RANGE,
    min_mass: float = BOOST_CLUSTER_MIN_MASS,
    cap: float = CLUSTER_EAT_MASS_CAP,
) -> float:
    """Bonus for boosting into a nearby high-mass pile or trail.

    Zero unless the step is closing (or already eating), the locked target is
    inside range_, and its mass clears min_mass. Scale is log-compressed.
    """
    if not closing or scale <= 0.0 or range_ <= 0.0:
        return 0.0
    if dist is None or dist >= range_ or mass < min_mass:
        return 0.0
    closeness = max(0.0, 1.0 - dist / range_)
    return scale * squash_mass(mass, cap) * closeness


def point_to_segment_dist(
    px: float, py: float, x0: float, y0: float, x1: float, y1: float
) -> float:
    vx, vy = x1 - x0, y1 - y0
    len_sq = vx * vx + vy * vy
    if len_sq < 1e-9:
        return math.hypot(px - x0, py - y0)
    t = max(0.0, min(1.0, ((px - x0) * vx + (py - y0) * vy) / len_sq))
    return math.hypot(px - (x0 + t * vx), py - (y0 + t * vy))


def vanished_foods(
    pre_foods: Iterable[FoodItem],
    post_foods: Iterable[FoodItem],
    match_radius: float = EAT_MATCH_RADIUS,
) -> List[FoodItem]:
    """Pre-step foods with no matching post-step food within match_radius."""
    post = [q for q in post_foods if q is not None and len(q) >= 2]
    if not post:
        return [p for p in pre_foods if p is not None and len(p) >= 2]
    cell = max(match_radius, 1.0)
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for j, q in enumerate(post):
        bx = int(math.floor(float(q[0]) / cell))
        by = int(math.floor(float(q[1]) / cell))
        buckets.setdefault((bx, by), []).append(j)
    used = [False] * len(post)
    match_sq = match_radius * match_radius
    gone: List[FoodItem] = []
    for p in pre_foods:
        if p is None or len(p) < 2:
            continue
        px, py = float(p[0]), float(p[1])
        gx = int(math.floor(px / cell))
        gy = int(math.floor(py / cell))
        best_j = -1
        best_d = match_sq
        for ix in (gx - 1, gx, gx + 1):
            for iy in (gy - 1, gy, gy + 1):
                for j in buckets.get((ix, iy), ()):
                    if used[j]:
                        continue
                    q = post[j]
                    dx = px - float(q[0])
                    dy = py - float(q[1])
                    d = dx * dx + dy * dy
                    if d <= best_d:
                        best_d = d
                        best_j = j
        if best_j >= 0:
            used[best_j] = True
        else:
            gone.append(p)
    return gone


def _near_segment(
    foods: Iterable[FoodItem], x0: float, y0: float, x1: float, y1: float, margin: float,
) -> List[FoodItem]:
    """Bounding-box prefilter: foods within `margin` of the segment's bbox.

    Cheap (4 comparisons/item) vs. the O(n) bucket-build downstream. A single
    step's segment is short, so this typically shrinks hundreds of candidates
    down to a handful before the real (still exact) distance checks run.
    """
    xmin = min(x0, x1) - margin
    xmax = max(x0, x1) + margin
    ymin = min(y0, y1) - margin
    ymax = max(y0, y1) + margin
    out = []
    for f in foods:
        if f is None or len(f) < 2:
            continue
        x, y = f[0], f[1]
        if xmin <= x <= xmax and ymin <= y <= ymax:
            out.append(f)
    return out


def eaten_food_mass(
    pre_foods: Iterable[FoodItem],
    post_foods: Iterable[FoodItem],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    eat_radius: float = 60.0,
    match_radius: float = EAT_MATCH_RADIUS,
) -> Tuple[float, int]:
    """Mass and count of vanished foods that sat on the snake's path this step.

    Only foods within eat_radius of the path segment can ever be counted, and
    only foods within match_radius beyond that can affect vanish-matching for
    them, so both food lists are bounding-box-filtered before the exact
    vanished_foods() pass instead of scanning every in-range pellet.
    """
    near_pre = _near_segment(pre_foods, x0, y0, x1, y1, eat_radius)
    if not near_pre:
        return 0.0, 0
    near_post = _near_segment(post_foods, x0, y0, x1, y1, eat_radius + match_radius)
    mass = 0.0
    count = 0
    for f in vanished_foods(near_pre, near_post, match_radius=match_radius):
        fx, fy = float(f[0]), float(f[1])
        if point_to_segment_dist(fx, fy, x0, y0, x1, y1) <= eat_radius:
            mass += food_size(f)
            count += 1
    return mass, count


def select_visible_foods(
    foods: Iterable[Tuple[float, float, float]],
    mx: float,
    my: float,
    sense_range: float = FOOD_SENSE_RANGE,
    max_foods: Optional[int] = None,
) -> List[List[float]]:
    """Keep foods within sense_range. If over max_foods, prefer mass/proximity."""
    if max_foods is None:
        max_foods = configured_max_foods()
    range_sq = sense_range * sense_range
    scored = []
    for item in foods:
        x, y, sz = item[0], item[1], item[2]
        dx = x - mx
        dy = y - my
        dist_sq = dx * dx + dy * dy
        if dist_sq > range_sq:
            continue
        dist = math.sqrt(dist_sq)
        score = sz / (dist + FOOD_KEEP_DIST_BIAS)
        scored.append((score, x, y, sz))
    if len(scored) > max_foods:
        scored.sort(key=lambda t: t[0], reverse=True)
        scored = scored[:max_foods]
    return [[x, y, sz] for _, x, y, sz in scored]


def _bin_foods(
    foods: Iterable[FoodItem],
    mx: float,
    my: float,
    cell: float,
    sense_range: float,
) -> Dict[Tuple[int, int], List]:
    """Map cell -> [mass, accx, accy, pellets]. pellets are (fx, fy, sz)."""
    range_sq = sense_range * sense_range
    bins: Dict[Tuple[int, int], List] = {}
    inv_cell = 1.0 / max(cell, 1e-6)
    for f in foods:
        if f is None or len(f) < 2:
            continue
        fx, fy = float(f[0]), float(f[1])
        sz = food_size(f)
        dx = fx - mx
        dy = fy - my
        dist_sq = dx * dx + dy * dy
        if dist_sq > range_sq:
            continue
        bx = int(math.floor(dx * inv_cell + 0.5))
        by = int(math.floor(dy * inv_cell + 0.5))
        rec = bins.get((bx, by))
        if rec is None:
            bins[(bx, by)] = [sz, fx * sz, fy * sz, [(fx, fy, sz)]]
        else:
            rec[0] += sz
            rec[1] += fx * sz
            rec[2] += fy * sz
            rec[3].append((fx, fy, sz))
    return bins


def cluster_foods(
    foods: Iterable[FoodItem],
    mx: float,
    my: float,
    cell: float = FOOD_CLUSTER_CELL,
    sense_range: float = FOOD_SENSE_RANGE,
) -> List[Tuple[float, float, float, float]]:
    """Bin foods into cells. Returns (cx, cy, dist, mass) per occupied cell."""
    out = []
    for mass, accx, accy, _pellets in _bin_foods(
        foods, mx, my, cell=cell, sense_range=sense_range
    ).values():
        if mass <= 0.0:
            continue
        cx = accx / mass
        cy = accy / mass
        dist = math.hypot(cx - mx, cy - my)
        out.append((cx, cy, dist, mass))
    return out


def _iter_food_strings(
    foods: Iterable[FoodItem],
    mx: float,
    my: float,
    cell: float = FOOD_CLUSTER_CELL,
    sense_range: float = FOOD_SENSE_RANGE,
) -> List[Tuple[float, float, float, float]]:
    """8-connected cell components. Aim at the nearest pellet on each string.

    Returns (aim_x, aim_y, aim_dist, total_mass) per component.
    """
    bins = _bin_foods(foods, mx, my, cell=cell, sense_range=sense_range)
    visited = set()
    strings: List[Tuple[float, float, float, float]] = []
    for start in bins:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        cells = []
        while stack:
            key = stack.pop()
            cells.append(key)
            bx, by = key
            for dx, dy in _NEIGHBOR_8:
                nb = (bx + dx, by + dy)
                if nb in bins and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        pellets: List[Tuple[float, float, float]] = []
        mass = 0.0
        for key in cells:
            rec = bins[key]
            mass += rec[0]
            pellets.extend(rec[3])
        if mass <= 0.0 or not pellets:
            continue
        nearest = min(pellets, key=lambda p: (p[0] - mx) ** 2 + (p[1] - my) ** 2)
        dist = math.hypot(nearest[0] - mx, nearest[1] - my)
        strings.append((nearest[0], nearest[1], dist, mass))
    return strings


def _best_from_clusters(
    clusters: Iterable[Tuple[float, float, float, float]],
    sense_range: float,
) -> Optional[Tuple[float, float, float, float]]:
    best = None
    best_score = -1.0
    inv = 1.0 / max(sense_range, 1.0)
    for cx, cy, dist, mass in clusters:
        score = mass * max(0.0, 1.0 - dist * inv)
        if score > best_score:
            best_score = score
            best = (cx, cy, dist, mass)
    return best


def best_food_target(
    foods: Iterable[FoodItem],
    mx: float,
    my: float,
    sense_range: float = FOOD_SENSE_RANGE,
    cell: float = FOOD_CLUSTER_CELL,
) -> Optional[Tuple[float, float, float, float]]:
    """Richest nearby string: maximize total_mass * (1 - nearest_dist/range).

    Aim point is the nearest pellet on that string, not the centroid.
    Returns (cx, cy, dist, mass) or None.
    """
    return _best_from_clusters(
        _iter_food_strings(foods, mx, my, cell=cell, sense_range=sense_range),
        sense_range,
    )


def locked_food_target(
    foods: Iterable[FoodItem],
    mx: float,
    my: float,
    locked: Optional[Tuple[float, float, float, float]] = None,
    sense_range: float = FOOD_SENSE_RANGE,
    cell: float = FOOD_CLUSTER_CELL,
    lock_radius: float = FOOD_LOCK_RADIUS,
) -> Optional[Tuple[float, float, float, float]]:
    """Keep the previous string while its aim point is still nearby; else pick a new best.

    The lock is the previous nearest pellet. Each step the aim slides to the
    new near end, which stays inside lock_radius as pellets are eaten.
    lock_radius <= 0 disables hysteresis and always returns best_food_target.
    """
    strings = _iter_food_strings(foods, mx, my, cell=cell, sense_range=sense_range)
    if locked is not None and lock_radius > 0:
        lx, ly = float(locked[0]), float(locked[1])
        best_match = None
        best_d = lock_radius
        for cx, cy, dist, mass in strings:
            d = math.hypot(cx - lx, cy - ly)
            if d <= best_d:
                best_d = d
                best_match = (cx, cy, dist, mass)
        if best_match is not None:
            return best_match
    return _best_from_clusters(strings, sense_range)


def idle_food_cost(
    cluster_dist: Optional[float],
    ate: bool,
    penalty: float,
    commit_range: float,
) -> float:
    """Flat miss cost while a food cluster is already in reach.

    Zero when the step ate, when no cluster is in range, or when the knobs
    are disabled. Does not replace starvation_penalty (long-horizon hunger).
    """
    if ate or penalty <= 0.0 or commit_range <= 0.0:
        return 0.0
    if cluster_dist is None:
        return 0.0
    if cluster_dist < commit_range:
        return penalty
    return 0.0


def js_collect_foods() -> str:
    """JS snippet: scan foods + preys out to FOOD_SENSE_RANGE, keep top MAX_FOODS.

    Expects JS locals: window.foods, window.preys, my_snake, MAX_FOODS.
    Clamped to FOOD_SENSE_RANGE so the Selenium backend matches websocket.
    Prunes the candidate list to MAX_FOODS whenever it hits
    MAX_FOODS * FOOD_LIST_PRUNE_MULT so a busy server does not full-sort
    every in-range pellet each step.
    Defines: visible_foods as [[x, y, sz], ...].
    """
    sense = "%.1f" % FOOD_SENSE_RANGE
    bias = "%.1f" % FOOD_KEEP_DIST_BIAS
    prey_sz = "%.1f" % PREY_DEFAULT_SIZE
    prune_mult = int(FOOD_LIST_PRUNE_MULT)
    return (
        "            var visible_foods = [];\n"
        "            var foodList = [];\n"
        "            var myX = my_snake.x, myY = my_snake.y;\n"
        f"            var senseRange = {sense};\n"
        "            var senseRangeSq = senseRange * senseRange;\n"
        f"            var pruneAt = MAX_FOODS * {prune_mult};\n"
        "            function considerFood(fx, fy, sz) {\n"
        "                var dx = fx - myX, dy = fy - myY;\n"
        "                var distSq = dx*dx + dy*dy;\n"
        "                if (distSq > senseRangeSq) return;\n"
        "                var dist = Math.sqrt(distSq);\n"
        f"                foodList.push([fx, fy, sz, sz / (dist + {bias})]);\n"
        "                if (foodList.length >= pruneAt) {\n"
        "                    foodList.sort(function(a, b) { return b[3] - a[3]; });\n"
        "                    foodList.length = MAX_FOODS;\n"
        "                }\n"
        "            }\n"
        "            if (window.foods && window.foods.length) {\n"
        "                for (var i = 0; i < window.foods.length; i++) {\n"
        "                    var f = window.foods[i];\n"
        "                    if (!f) continue;\n"
        "                    var fx = (typeof f.xx === 'number') ? f.xx : (typeof f.x === 'number') ? f.x : (typeof f.rx === 'number') ? f.rx : null;\n"
        "                    var fy = (typeof f.yy === 'number') ? f.yy : (typeof f.y === 'number') ? f.y : (typeof f.ry === 'number') ? f.ry : null;\n"
        "                    if (fx === null || fy === null) continue;\n"
        "                    considerFood(fx, fy, f.sz || 1);\n"
        "                }\n"
        "            }\n"
        "            if (window.preys && window.preys.length) {\n"
        "                for (var i = 0; i < window.preys.length; i++) {\n"
        "                    var p = window.preys[i];\n"
        "                    if (!p) continue;\n"
        "                    var px = (typeof p.xx === 'number') ? p.xx : (typeof p.x === 'number') ? p.x : (typeof p.rx === 'number') ? p.rx : null;\n"
        "                    var py = (typeof p.yy === 'number') ? p.yy : (typeof p.y === 'number') ? p.y : (typeof p.ry === 'number') ? p.ry : null;\n"
        "                    if (px === null || py === null) continue;\n"
        f"                    considerFood(px, py, p.sz || {prey_sz});\n"
        "                }\n"
        "            }\n"
        "            if (foodList.length > MAX_FOODS) {\n"
        "                foodList.sort(function(a, b) { return b[3] - a[3]; });\n"
        "                foodList.length = MAX_FOODS;\n"
        "            }\n"
        "            for (var i = 0; i < foodList.length; i++)\n"
        "                visible_foods.push([foodList[i][0], foodList[i][1], foodList[i][2]]);\n"
    )
