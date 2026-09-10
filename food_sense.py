"""Food sensing: range, density, and cluster targeting.

The game camera is often ~400 units at spawn while the observation matrix
covers 1000 and sectors cover 2000. Food must be collected out to
FOOD_SENSE_RANGE, and density (not nearest-pellet) is what the agent uses.
"""

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

FOOD_SENSE_RANGE = 2000.0
FOOD_CLUSTER_CELL = 120.0
MAX_FOODS = 800
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


def food_draw_radius_px(sz: float, scale: float) -> float:
    """Matrix-pixel radius so large pellets occupy more cells than crumbs."""
    return max(FOOD_DRAW_RADIUS_MIN_PX, float(sz) * FOOD_DRAW_RADIUS_PER_SZ * float(scale))


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
    """Mass and count of vanished foods that sat on the snake's path this step."""
    mass = 0.0
    count = 0
    for f in vanished_foods(pre_foods, post_foods, match_radius=match_radius):
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
    max_foods: int = MAX_FOODS,
) -> List[List[float]]:
    """Keep foods within sense_range. If over max_foods, prefer mass/proximity."""
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


def cluster_foods(
    foods: Iterable[FoodItem],
    mx: float,
    my: float,
    cell: float = FOOD_CLUSTER_CELL,
    sense_range: float = FOOD_SENSE_RANGE,
) -> List[Tuple[float, float, float, float]]:
    """Bin foods into cells. Returns (cx, cy, dist, mass) per occupied cell."""
    range_sq = sense_range * sense_range
    bins: Dict[Tuple[int, int], List[float]] = {}
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
        bx = int(math.floor(dx / cell + 0.5))
        by = int(math.floor(dy / cell + 0.5))
        rec = bins.get((bx, by))
        if rec is None:
            bins[(bx, by)] = [sz, fx * sz, fy * sz]
        else:
            rec[0] += sz
            rec[1] += fx * sz
            rec[2] += fy * sz
    out = []
    for mass, accx, accy in bins.values():
        if mass <= 0.0:
            continue
        cx = accx / mass
        cy = accy / mass
        dist = math.hypot(cx - mx, cy - my)
        out.append((cx, cy, dist, mass))
    return out


def best_food_target(
    foods: Iterable[FoodItem],
    mx: float,
    my: float,
    sense_range: float = FOOD_SENSE_RANGE,
    cell: float = FOOD_CLUSTER_CELL,
) -> Optional[Tuple[float, float, float, float]]:
    """Densest nearby cluster: maximize mass * (1 - dist/range).

    Returns (cx, cy, dist, mass) or None.
    """
    best = None
    best_score = -1.0
    inv = 1.0 / max(sense_range, 1.0)
    for cx, cy, dist, mass in cluster_foods(
        foods, mx, my, cell=cell, sense_range=sense_range
    ):
        score = mass * max(0.0, 1.0 - dist * inv)
        if score > best_score:
            best_score = score
            best = (cx, cy, dist, mass)
    return best


def js_collect_foods() -> str:
    """JS snippet: scan foods + preys out to FOOD_SENSE_RANGE, keep top MAX_FOODS.

    Expects JS locals: window.foods, window.preys, my_snake, viewRadius, MAX_FOODS.
    Defines: visible_foods as [[x, y, sz], ...].
    """
    sense = "%.1f" % FOOD_SENSE_RANGE
    bias = "%.1f" % FOOD_KEEP_DIST_BIAS
    prey_sz = "%.1f" % PREY_DEFAULT_SIZE
    return (
        "            var visible_foods = [];\n"
        "            var foodList = [];\n"
        "            var myX = my_snake.x, myY = my_snake.y;\n"
        f"            var senseRange = Math.max(viewRadius * 1.2, {sense});\n"
        "            var senseRangeSq = senseRange * senseRange;\n"
        "            if (window.foods && window.foods.length) {\n"
        "                for (var i = 0; i < window.foods.length; i++) {\n"
        "                    var f = window.foods[i];\n"
        "                    if (!f) continue;\n"
        "                    var fx = (typeof f.xx === 'number') ? f.xx : (typeof f.x === 'number') ? f.x : (typeof f.rx === 'number') ? f.rx : null;\n"
        "                    var fy = (typeof f.yy === 'number') ? f.yy : (typeof f.y === 'number') ? f.y : (typeof f.ry === 'number') ? f.ry : null;\n"
        "                    if (fx === null || fy === null) continue;\n"
        "                    var dx = fx - myX, dy = fy - myY;\n"
        "                    var distSq = dx*dx + dy*dy;\n"
        "                    if (distSq > senseRangeSq) continue;\n"
        "                    var sz = f.sz || 1;\n"
        "                    var dist = Math.sqrt(distSq);\n"
        f"                    foodList.push([fx, fy, sz, sz / (dist + {bias})]);\n"
        "                }\n"
        "            }\n"
        "            if (window.preys && window.preys.length) {\n"
        "                for (var i = 0; i < window.preys.length; i++) {\n"
        "                    var p = window.preys[i];\n"
        "                    if (!p) continue;\n"
        "                    var px = (typeof p.xx === 'number') ? p.xx : (typeof p.x === 'number') ? p.x : (typeof p.rx === 'number') ? p.rx : null;\n"
        "                    var py = (typeof p.yy === 'number') ? p.yy : (typeof p.y === 'number') ? p.y : (typeof p.ry === 'number') ? p.ry : null;\n"
        "                    if (px === null || py === null) continue;\n"
        "                    var dx = px - myX, dy = py - myY;\n"
        "                    var distSq = dx*dx + dy*dy;\n"
        "                    if (distSq > senseRangeSq) continue;\n"
        f"                    var sz = p.sz || {prey_sz};\n"
        "                    var dist = Math.sqrt(distSq);\n"
        f"                    foodList.push([px, py, sz, sz / (dist + {bias})]);\n"
        "                }\n"
        "            }\n"
        "            foodList.sort(function(a, b) { return b[3] - a[3]; });\n"
        "            for (var i = 0; i < Math.min(foodList.length, MAX_FOODS); i++)\n"
        "                visible_foods.push([foodList[i][0], foodList[i][1], foodList[i][2]]);\n"
    )
