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
# Per-pixel food-channel saturation. Overlapping pellets add their sz, so a
# dead snake's remains (many sz 5-12 pellets a body-point apart) sum well past
# 20 per pixel; 40 keeps a brightness gradient between a medium and a large
# pile while a lone sz=1 crumb still paints ~47/255.
FOOD_CHANNEL_LOG_CAP = 40.0
# Distance-weighted per-sector score (legacy band).
FOOD_SECTOR_LOG_CAP = 12.0
# Unweighted per-sector mass: ~500 is a big snake's remains in one 15° slice,
# ordinary background pellets sum to a few tens.
FOOD_SECTOR_MASS_CAP = 500.0
# Total pellet mass inside the 2000-unit sense disc.
FOOD_TOTAL_LOG_CAP = 3000.0
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


def squash_mass_array(mass, cap: float) -> np.ndarray:
    """Vector form of squash_mass(): (0, 1] log-compressed, 0 for mass <= 0."""
    m = np.asarray(mass, dtype=np.float64)
    if cap <= 0.0:
        return np.zeros(m.shape, dtype=np.float64)
    out = np.log1p(np.maximum(m, 0.0)) / math.log1p(cap)
    return np.minimum(out, 1.0)


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


def foods_xyz(foods: Iterable[FoodItem]) -> np.ndarray:
    """Foods -> (n, 3) float64 rows [x, y, food_size(item)].

    An (n, 3) float64 array is assumed to be in this format already and is
    returned as-is, so callers can convert once per tick and pass the array
    to every consumer. Other numeric arrays are cast in place; the usual
    homogeneous [x, y, sz] rows convert with one np.asarray; ragged or
    malformed lists fall back to per-item food_size().
    """
    if isinstance(foods, np.ndarray):
        if foods.ndim == 2 and foods.shape[1] == 3 and foods.dtype == np.float64:
            return foods
        if foods.size == 0:
            return np.empty((0, 3), dtype=np.float64)
    elif not foods:
        return np.empty((0, 3), dtype=np.float64)
    try:
        raw = np.asarray(foods, dtype=np.float64)
    except (TypeError, ValueError):
        raw = None
    if raw is not None and raw.ndim == 2 and raw.shape[1] >= 2:
        out = np.empty((raw.shape[0], 3), dtype=np.float64)
        out[:, :2] = raw[:, :2]
        if raw.shape[1] > 2:
            sz = raw[:, 2]
            out[:, 2] = np.where(sz > 0.0, sz, 1.0)  # food_size(): non-positive -> 1.0
        else:
            out[:, 2] = 1.0
        return out
    return np.asarray(
        [(f[0], f[1], food_size(f)) for f in foods if f is not None and len(f) >= 2],
        dtype=np.float64,
    ).reshape(-1, 3)


def _in_range(foods, mx, my, sense_range):
    """Foods within sense_range -> (fx, fy, sz, dx, dy, d2) float64 arrays."""
    arr = foods_xyz(foods)
    fx, fy, sz = arr[:, 0], arr[:, 1], arr[:, 2]
    dx = fx - mx
    dy = fy - my
    d2 = dx * dx + dy * dy
    keep = d2 <= sense_range * sense_range
    if not keep.all():
        fx, fy, sz, dx, dy, d2 = (a[keep] for a in (fx, fy, sz, dx, dy, d2))
    return fx, fy, sz, dx, dy, d2


def _cell_groups(dx, dy, cell):
    """Group in-range foods by cell.

    Returns (bx, by, first, inv): per-food cell coords, the first food index
    of each cell (cells numbered in first-occurrence order, the same order a
    dict of cells would iterate in), and each food's cell number.
    """
    inv_cell = 1.0 / max(cell, 1e-6)
    bx = np.floor(dx * inv_cell + 0.5).astype(np.int64)
    by = np.floor(dy * inv_cell + 0.5).astype(np.int64)
    if bx.size == 0:
        return bx, by, np.empty(0, np.int64), np.empty(0, np.int64)
    key = (bx - bx.min()) * (by.max() - by.min() + 1) + (by - by.min())
    _uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
    order = np.argsort(first, kind='stable')  # np.unique sorted by key value
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return bx, by, first[order], rank[inv]


def _cell_components(cx, cy):
    """8-connected components of occupied cells -> per-cell label.

    Pure numpy: min-label hooking over the neighbour edges plus pointer
    jumping until stable. Each component ends up labelled with its lowest
    cell index, so sorting labels orders components by first cell.
    """
    k = cx.size
    label = np.arange(k)
    if k < 2:
        return label
    ox = cx - cx.min()
    oy = cy - cy.min()
    w = int(ox.max()) + 1
    h = int(oy.max()) + 1
    key = oy * w + ox
    order = np.argsort(key)
    skey = key[order]
    us = []
    vs = []
    # Forward half of the 8-neighbourhood; each undirected edge found once.
    for ddx, ddy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        nx = ox + ddx
        ny = oy + ddy
        ok = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
        nkey = ny[ok] * w + nx[ok]
        pos = np.minimum(np.searchsorted(skey, nkey), k - 1)
        hit = skey[pos] == nkey
        us.append(np.flatnonzero(ok)[hit])
        vs.append(order[pos[hit]])
    u = np.concatenate(us)
    v = np.concatenate(vs)
    if u.size == 0:
        return label
    while True:
        prev = label
        label = label.copy()
        m = np.minimum(label[u], label[v])
        np.minimum.at(label, u, m)
        np.minimum.at(label, v, m)
        label = label[label]
        if np.array_equal(label, prev):
            return label


def cluster_foods(
    foods: Iterable[FoodItem],
    mx: float,
    my: float,
    cell: float = FOOD_CLUSTER_CELL,
    sense_range: float = FOOD_SENSE_RANGE,
) -> List[Tuple[float, float, float, float]]:
    """Bin foods into cells. Returns (cx, cy, dist, mass) per occupied cell."""
    fx, fy, sz, dx, dy, _d2 = _in_range(foods, mx, my, sense_range)
    if fx.size == 0:
        return []
    _bx, _by, first, inv = _cell_groups(dx, dy, cell)
    k = first.size
    mass = np.bincount(inv, weights=sz, minlength=k)
    accx = np.bincount(inv, weights=fx * sz, minlength=k)
    accy = np.bincount(inv, weights=fy * sz, minlength=k)
    out = []
    for i in range(k):
        m = float(mass[i])
        if m <= 0.0:
            continue
        cx = float(accx[i]) / m
        cy = float(accy[i]) / m
        out.append((cx, cy, math.hypot(cx - mx, cy - my), m))
    return out


def _iter_food_strings(
    foods: Iterable[FoodItem],
    mx: float,
    my: float,
    cell: float = FOOD_CLUSTER_CELL,
    sense_range: float = FOOD_SENSE_RANGE,
) -> List[Tuple[float, float, float, float]]:
    """8-connected cell components. Aim at the nearest pellet on each string.

    Returns (aim_x, aim_y, aim_dist, total_mass) per component, components
    ordered by their first pellet. Nearest-pellet ties go to the lowest index.
    """
    fx, fy, sz, dx, dy, d2 = _in_range(foods, mx, my, sense_range)
    if fx.size == 0:
        return []
    bx, by, first, inv = _cell_groups(dx, dy, cell)
    label = _cell_components(bx[first], by[first])
    _reps, cidx = np.unique(label, return_inverse=True)  # sorted = first-cell order
    comp = cidx[inv]  # per food
    n_comp = _reps.size
    mass = np.bincount(comp, weights=sz, minlength=n_comp)
    order = np.lexsort((d2, comp))
    comp_sorted = comp[order]
    starts = np.flatnonzero(np.r_[True, comp_sorted[1:] != comp_sorted[:-1]])
    nearest = order[starts]  # one food index per component, in component order
    strings: List[Tuple[float, float, float, float]] = []
    for ci in range(n_comp):
        m = float(mass[ci])
        if m <= 0.0:
            continue
        i = nearest[ci]
        nx = float(fx[i])
        ny = float(fy[i])
        strings.append((nx, ny, math.hypot(nx - mx, ny - my), m))
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
