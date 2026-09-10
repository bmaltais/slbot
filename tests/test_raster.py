"""Vectorized observation paths vs the per-item loops they replaced.

The reference implementations below are the pre-vectorization code
(per-pixel disc loop, per-segment circle interpolation, dict-based food
binning + flood fill). The vectorized versions must reproduce them.
"""
import math
import os
import random
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import food_sense
from slither_env import (
    FRAME_DTYPE,
    SlitherEnv,
    _disc_pixels,
    _thick_polyline_samples,
    quantize_frame,
)
from worker_session import spawning_obs

SIZE = 40


# --- reference: old per-pixel primitives -------------------------------------

def ref_draw_circle(matrix, channel, cx, cy, r, value, blend='set'):
    size = matrix.shape[1]
    r_int = int(math.ceil(r))
    x_min = max(0, int(cx - r_int))
    x_max = min(size, int(cx + r_int + 1))
    y_min = max(0, int(cy - r_int))
    y_max = min(size, int(cy + r_int + 1))
    r_sq = r * r
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            dx = x - cx
            dy = y - cy
            if dx * dx + dy * dy <= r_sq:
                if blend == 'max':
                    if value > matrix[channel, y, x]:
                        matrix[channel, y, x] = value
                elif blend == 'add':
                    matrix[channel, y, x] += value
                else:
                    matrix[channel, y, x] = value


def ref_thick_line_samples(x0, y0, x1, y1, radius):
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist == 0:
        return [(x0, y0)]
    steps = int(dist / max(0.5, radius * 0.5)) + 1
    out = []
    for i in range(steps + 1):
        t = i / max(1, steps)
        out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return out


def ref_thick_line(matrix, channel, x0, y0, x1, y1, width, value):
    for x, y in ref_thick_line_samples(x0, y0, x1, y1, width / 2.0):
        ref_draw_circle(matrix, channel, x, y, width / 2.0, value)


# --- reference: old dict-based food strings -----------------------------------

_NEIGHBOR_8 = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy)


def ref_bin_foods(foods, mx, my, cell, sense_range):
    range_sq = sense_range * sense_range
    bins = {}
    inv_cell = 1.0 / max(cell, 1e-6)
    for f in foods:
        if f is None or len(f) < 2:
            continue
        fx, fy = float(f[0]), float(f[1])
        sz = food_sense.food_size(f)
        dx, dy = fx - mx, fy - my
        if dx * dx + dy * dy > range_sq:
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


def ref_cluster_foods(foods, mx, my, cell, sense_range):
    out = []
    for mass, accx, accy, _ in ref_bin_foods(foods, mx, my, cell, sense_range).values():
        cx, cy = accx / mass, accy / mass
        out.append((cx, cy, math.hypot(cx - mx, cy - my), mass))
    return out


def ref_food_strings(foods, mx, my, cell, sense_range):
    bins = ref_bin_foods(foods, mx, my, cell, sense_range)
    visited = set()
    strings = []
    for start in bins:
        if start in visited:
            continue
        stack, cells = [start], []
        visited.add(start)
        while stack:
            key = stack.pop()
            cells.append(key)
            for dx, dy in _NEIGHBOR_8:
                nb = (key[0] + dx, key[1] + dy)
                if nb in bins and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        pellets, mass = [], 0.0
        for key in cells:
            mass += bins[key][0]
            pellets.extend(bins[key][3])
        nearest = min(pellets, key=lambda p: (p[0] - mx) ** 2 + (p[1] - my) ** 2)
        strings.append((nearest[0], nearest[1], math.hypot(nearest[0] - mx, nearest[1] - my), mass))
    return strings


def _make_env(size=SIZE):
    with patch('slither_env._create_browser', return_value=MagicMock()):
        env = SlitherEnv(headless=True, nickname="T", matrix_size=size, view_plus=False)
    env.browser = MagicMock()
    return env


def _random_discs(rng, n, size=SIZE):
    cx = rng.uniform(-6.0, size + 6.0, n)
    cy = rng.uniform(-6.0, size + 6.0, n)
    r = np.concatenate([rng.uniform(0.0, 6.0, n - 3), [0.0, 1.0, 2.5]])
    return cx, cy, r


class TestDiscRaster(unittest.TestCase):
    def setUp(self):
        self.env = _make_env()
        self.rng = np.random.default_rng(42)

    def test_disc_pixels_match_reference_loop_per_disc(self):
        cx, cy, r = _random_discs(self.rng, 300)
        for i in range(cx.size):
            ref = np.zeros((1, SIZE, SIZE), dtype=np.float32)
            ref_draw_circle(ref, 0, float(cx[i]), float(cy[i]), float(r[i]), 1.0)
            ys, xs, di = _disc_pixels(cx[i], cy[i], r[i], SIZE)
            got = np.zeros((SIZE, SIZE), dtype=np.float32)
            got[ys, xs] = 1.0
            np.testing.assert_array_equal(got, ref[0], err_msg=f"disc {i}: c=({cx[i]}, {cy[i]}) r={r[i]}")
            self.assertTrue(np.all(di == 0))

    def test_batched_set_and_max_match_sequential_reference(self):
        cx, cy, r = _random_discs(self.rng, 200)
        vals = self.rng.uniform(0.1, 1.0, cx.size)
        for blend in ('set', 'max'):
            ref = np.zeros((1, SIZE, SIZE), dtype=np.float32)
            got = np.zeros((1, SIZE, SIZE), dtype=np.float32)
            if blend == 'set':
                for i in range(cx.size):
                    ref_draw_circle(ref, 0, cx[i], cy[i], r[i], 0.5, blend)
                self.env._draw_discs(got, 0, cx, cy, r, 0.5, blend=blend)
            else:
                for i in range(cx.size):
                    ref_draw_circle(ref, 0, cx[i], cy[i], r[i], vals[i], blend)
                self.env._draw_discs(got, 0, cx, cy, r, vals, blend=blend)
            np.testing.assert_array_equal(got, ref, err_msg=blend)

    def test_batched_add_matches_sequential_reference(self):
        cx, cy, r = _random_discs(self.rng, 200)
        vals = self.rng.uniform(0.5, 12.0, cx.size)
        ref = np.zeros((1, SIZE, SIZE), dtype=np.float32)
        for i in range(cx.size):
            ref_draw_circle(ref, 0, cx[i], cy[i], r[i], vals[i], 'add')
        got = np.zeros((1, SIZE, SIZE), dtype=np.float32)
        self.env._draw_discs(got, 0, cx, cy, r, vals, blend='add')
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)

    def test_set_rejects_per_disc_values(self):
        m = np.zeros((1, SIZE, SIZE), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.env._draw_discs(m, 0, [1.0, 2.0], [1.0, 2.0], 1.0, [0.5, 1.0], blend='set')

    def test_fully_offscreen_and_empty_inputs_draw_nothing(self):
        m = np.zeros((1, SIZE, SIZE), dtype=np.float32)
        self.env._draw_discs(m, 0, [-50.0, SIZE + 50.0], [10.0, 10.0], 3.0, 1.0)
        self.env._draw_discs(m, 0, [], [], 3.0, 1.0)
        self.assertEqual(float(m.sum()), 0.0)


class TestPolyline(unittest.TestCase):
    def test_samples_match_reference_per_segment(self):
        rng = np.random.default_rng(7)
        for radius in (1.25, 2.32, 0.4, 7.0):
            px = rng.uniform(-10, SIZE + 10, 12)
            py = rng.uniform(-10, SIZE + 10, 12)
            ref = []
            for i in range(px.size - 1):
                ref.extend(ref_thick_line_samples(px[i], py[i], px[i + 1], py[i + 1], radius))
            sx, sy = _thick_polyline_samples(px, py, radius)
            ref = np.asarray(ref)
            np.testing.assert_array_equal(sx, ref[:, 0])
            np.testing.assert_array_equal(sy, ref[:, 1])

    def test_zero_length_segment_draws_the_point(self):
        sx, sy = _thick_polyline_samples([3.0, 3.0], [4.0, 4.0], 1.0)
        self.assertTrue(np.all(sx == 3.0) and np.all(sy == 4.0))
        self.assertGreaterEqual(sx.size, 1)

    def test_single_point_polyline(self):
        sx, sy = _thick_polyline_samples([5.0], [6.0], 1.0)
        self.assertEqual((sx.tolist(), sy.tolist()), ([5.0], [6.0]))

    def test_thick_polyline_matches_reference_segment_draws(self):
        env = _make_env()
        rng = np.random.default_rng(3)
        px = rng.uniform(0, SIZE, 8)
        py = rng.uniform(0, SIZE, 8)
        ref = np.zeros((1, SIZE, SIZE), dtype=np.float32)
        for i in range(px.size - 1):
            ref_thick_line(ref, 0, px[i], py[i], px[i + 1], py[i + 1], 4.6, 0.5)
        got = np.zeros((1, SIZE, SIZE), dtype=np.float32)
        env._draw_thick_polyline(got, 0, px, py, 4.6, 0.5)
        np.testing.assert_array_equal(got, ref)


def _scene(mx=21600.0, my=21600.0, ang=0.3, foods=(), enemies=(), my_pts=(), sc=1.0):
    return {
        'dead': False, 'valid': True,
        'self': {'x': mx, 'y': my, 'ang': ang, 'len': 30, 'sc': sc, 'sp': 5.7, 'pts': list(my_pts)},
        'foods': list(foods), 'enemies': list(enemies),
        'map_radius': 21600, 'map_center_x': 21600, 'map_center_y': 21600,
        'view_radius': 500, 'gsc': 1.0,
    }


def _ref_snake_channels(env, data):
    """Old draw order for channels 1 (enemies) and 2 (self), no wall."""
    size = env.matrix_size
    m = np.zeros((3, size, size), dtype=np.float32)
    me = data['self']
    mx, my, ang = me['x'], me['y'], me['ang']
    sin_a, cos_a = math.sin(ang), math.cos(ang)

    def ego_raw(dx, dy):
        return -sin_a * dx + cos_a * dy, -cos_a * dx - sin_a * dy

    def ego(dx, dy):
        rx, ry = ego_raw(dx, dy)
        return int(rx * env.scale + size / 2), int(ry * env.scale + size / 2)

    cg = size / 2
    for e in data['enemies']:
        rx, ry = ego_raw(e['x'] - mx, e['y'] - my)
        hx, hy = cg + rx * env.scale, cg + ry * env.scale
        w = max(2.5, e.get('sc', 1.0) * 29.0 * env.scale)
        if -50 < hx < size + 50 and -50 < hy < size + 50:
            ref_draw_circle(m, 1, hx, hy, (w / 2.0) * 1.2, 1.0)
        px, py = hx, hy
        for pt in e.get('pts', []):
            rx, ry = ego_raw(pt[0] - mx, pt[1] - my)
            gx, gy = cg + rx * env.scale, cg + ry * env.scale
            ref_thick_line(m, 1, px, py, gx, gy, w, 0.5)
            px, py = gx, gy
    cx, cy = size // 2, size // 2
    pts = me.get('pts', [])
    w = max(2.5, me.get('sc', 1.0) * 29.0 * env.scale)
    if pts:
        bx, by = ego(pts[0][0] - mx, pts[0][1] - my)
        ref_thick_line(m, 2, cx, cy, bx, by, w, 0.5)
    for i in range(len(pts) - 1):
        x1, y1 = ego(pts[i][0] - mx, pts[i][1] - my)
        x2, y2 = ego(pts[i + 1][0] - mx, pts[i + 1][1] - my)
        ref_thick_line(m, 2, x1, y1, x2, y2, w, 0.5)
    ref_draw_circle(m, 2, cx, cy, (w / 2.0) * 1.2, 1.0)
    return quantize_frame(m)


def _random_snake_scene(rng, n_enemies, n_pts, my_pts):
    mx, my = 21600.0, 21600.0
    enemies = []
    for _ in range(n_enemies):
        ex, ey = mx + rng.uniform(-900, 900), my + rng.uniform(-900, 900)
        h = rng.uniform(0, 6.3)
        pts, x, y = [], ex, ey
        sc = rng.choice([0.8, 1.0, 2.0])
        for _ in range(n_pts):
            h += rng.uniform(-0.5, 0.5)
            x -= 42 * sc * math.cos(h)
            y -= 42 * sc * math.sin(h)
            pts.append([x, y])
        enemies.append({'x': ex, 'y': ey, 'sc': sc, 'ang': h, 'sp': 5.7, 'pts': pts})
    mine, x, y, h = [], mx, my, rng.uniform(0, 6.3)
    for _ in range(my_pts):
        h += rng.uniform(-0.5, 0.5)
        x += 42 * math.cos(h)
        y += 42 * math.sin(h)
        mine.append([x, y])
    return _scene(mx, my, rng.uniform(-3.1, 3.1), enemies=enemies, my_pts=mine, sc=rng.choice([1.0, 1.6]))


class TestFrameRender(unittest.TestCase):
    def setUp(self):
        self.env = _make_env(size=48)

    def test_frame_is_uint8(self):
        m = self.env._process_data_to_matrix(_scene())
        self.assertEqual(m.dtype, FRAME_DTYPE)
        self.assertEqual(m.shape, (3, 48, 48))
        self.assertEqual(m[2, 24, 24], 255)  # own head

    def test_dead_or_empty_frames_are_zero_uint8(self):
        for data in ({}, None, {'dead': True}, {'dead': False}):
            m = self.env._process_data_to_matrix(data)
            self.assertEqual(m.dtype, FRAME_DTYPE)
            self.assertEqual(int(m.sum()), 0)
        self.assertEqual(self.env._matrix_zeros()['matrix'].dtype, FRAME_DTYPE)

    def test_enemy_head_and_body_values(self):
        # enemy 300 units straight ahead; a bare head paints 1.0 -> 255
        e = {'x': 21900.0, 'y': 21600.0, 'sc': 1.0, 'ang': 0.0, 'sp': 5.7, 'pts': []}
        m = self.env._process_data_to_matrix(_scene(ang=0.0, enemies=[e]))
        self.assertEqual(int(m[1].max()), 255)
        # body segments paint 0.5 -> 127 (and, drawn after the head, cover it)
        e['pts'] = [[22000.0, 21600.0], [22100.0, 21600.0]]
        m = self.env._process_data_to_matrix(_scene(ang=0.0, enemies=[e]))
        self.assertIn(127, np.unique(m[1]))

    def test_snake_channels_match_reference_draw_order(self):
        rng = random.Random(11)
        for i in range(12):
            data = _random_snake_scene(rng, rng.randint(0, 3), rng.randint(0, 25), rng.randint(0, 20))
            got = self.env._process_data_to_matrix(data)
            ref = _ref_snake_channels(self.env, data)
            np.testing.assert_array_equal(got[1:], ref[1:], err_msg=f"scene {i}")

    def test_ragged_body_points_tolerated(self):
        e = {'x': 21700.0, 'y': 21600.0, 'sc': 1.0, 'pts': [[21750.0, 21600.0], [21800.0], None, [21850.0, 21600.0, 9]]}
        m = self.env._process_data_to_matrix(_scene(enemies=[e]))
        self.assertEqual(int(m[1].max()), 255)


class TestQuantize(unittest.TestCase):
    def test_clips_and_scales(self):
        q = quantize_frame(np.array([-0.5, 0.0, 0.5, 1.0, 1.7], dtype=np.float32))
        np.testing.assert_array_equal(q, np.array([0, 0, 127, 255, 255], dtype=np.uint8))

    def test_uint8_passthrough_is_same_object(self):
        a = np.arange(6, dtype=np.uint8)
        self.assertIs(quantize_frame(a), a)


class TestFoodStrings(unittest.TestCase):
    def _foods(self, rng, n_trails, n_scatter):
        foods = []
        for _ in range(n_scatter):
            foods.append([rng.uniform(-2200, 2200), rng.uniform(-2200, 2200), rng.choice([1.0, 2.0, 4.0])])
        for _ in range(n_trails):
            x, y, h = rng.uniform(-1500, 1500), rng.uniform(-1500, 1500), rng.uniform(0, 6.3)
            for _ in range(rng.randint(1, 40)):
                h += rng.uniform(-0.6, 0.6)
                x += 55 * math.cos(h)
                y += 55 * math.sin(h)
                foods.append([x, y, rng.choice([1.0, 3.0])])
        return foods

    def test_strings_and_clusters_match_reference(self):
        rng = random.Random(5)
        for i in range(40):
            foods = self._foods(rng, rng.randint(0, 8), rng.randint(0, 300))
            mx, my = rng.uniform(-100, 100), rng.uniform(-100, 100)
            for fn, ref in ((food_sense._iter_food_strings, ref_food_strings),
                            (food_sense.cluster_foods, ref_cluster_foods)):
                got = fn(foods, mx, my, cell=120.0, sense_range=2000.0)
                exp = ref(foods, mx, my, 120.0, 2000.0)
                self.assertEqual(len(got), len(exp), f"scene {i} {fn.__name__}")
                np.testing.assert_allclose(np.asarray(got).reshape(-1, 4), np.asarray(exp).reshape(-1, 4),
                                           rtol=1e-12, atol=1e-9, err_msg=f"scene {i} {fn.__name__}")

    def test_array_input_matches_list_input(self):
        rng = random.Random(9)
        foods = self._foods(rng, 3, 50)
        arr = food_sense.foods_xyz(foods)
        self.assertIs(food_sense.foods_xyz(arr), arr)
        a = food_sense.locked_food_target(foods, 0.0, 0.0, None)
        b = food_sense.locked_food_target(arr, 0.0, 0.0, None)
        np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_foods_xyz_handles_ragged_and_odd_sizes(self):
        arr = food_sense.foods_xyz([[1.0, 2.0], [3.0, 4.0, 0.0], None, [5.0], [6.0, 7.0, 2.5]])
        np.testing.assert_array_equal(arr, [[1.0, 2.0, 1.0], [3.0, 4.0, 1.0], [6.0, 7.0, 2.5]])
        self.assertEqual(food_sense.foods_xyz([]).shape, (0, 3))
        homogeneous = food_sense.foods_xyz([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]])
        np.testing.assert_array_equal(homogeneous[:, 2], [1.0, 5.0])

    def test_empty_and_single(self):
        self.assertEqual(food_sense._iter_food_strings([], 0.0, 0.0), [])
        got = food_sense._iter_food_strings([[10.0, 0.0, 2.0]], 0.0, 0.0)
        self.assertEqual(len(got), 1)
        self.assertAlmostEqual(got[0][2], 10.0)
        self.assertAlmostEqual(got[0][3], 2.0)

    def test_components_join_diagonal_cells(self):
        # cells (0,0) and (1,1) touch diagonally: one string with both masses
        foods = [[0.0, 0.0, 1.0], [120.0, 120.0, 3.0], [600.0, 0.0, 1.0]]
        got = food_sense._iter_food_strings(foods, 0.0, 0.0, cell=120.0, sense_range=2000.0)
        self.assertEqual(len(got), 2)
        self.assertAlmostEqual(got[0][3], 4.0)


class TestSectorsAndDistances(unittest.TestCase):
    def setUp(self):
        self.env = _make_env()

    def test_body_point_scores_sector_as_body(self):
        e = {'x': 23500.0, 'y': 21600.0, 'sc': 1.0, 'ang': 0.0, 'sp': 5.7,
             'pts': [[22100.0, 21600.0]]}  # head out of scope, body 500 ahead
        s = self.env._compute_sectors(_scene(ang=0.0, enemies=[e]))
        expected = 1.0 - (500.0 - 14.5) / 2000.0
        self.assertAlmostEqual(float(s[24]), expected, places=5)
        self.assertEqual(float(s[48]), 0.0)
        self.assertTrue(np.all(s[49:72] == -1.0))

    def test_closer_head_keeps_head_type_over_body(self):
        e = {'x': 21900.0, 'y': 21600.0, 'sc': 1.0, 'ang': math.pi, 'sp': 5.7,
             'pts': [[22100.0, 21600.0]]}
        s = self.env._compute_sectors(_scene(ang=0.0, enemies=[e]))
        self.assertEqual(float(s[48]), 1.0)
        self.assertAlmostEqual(float(s[24]), 1.0 - (300.0 - 14.5) / 2000.0, places=5)

    def test_min_enemy_distance(self):
        e = {'x': 21900.0, 'y': 21600.0, 'sc': 1.0, 'pts': [[21700.0, 21600.0], [21605.0, 21600.0]]}
        self.assertEqual(self.env._min_enemy_distance([e], 21600.0, 21600.0), 0.0)
        e['pts'] = [[21700.0, 21600.0]]
        self.assertAlmostEqual(self.env._min_enemy_distance([e], 21600.0, 21600.0), 100.0 - 14.5)
        self.assertEqual(self.env._min_enemy_distance([], 0.0, 0.0), float('inf'))


class TestUint8Transport(unittest.TestCase):
    def test_placeholder_observations_are_uint8(self):
        self.assertEqual(spawning_obs(8)['matrix'].dtype, np.uint8)

    def test_vecframestack_keeps_uint8(self):
        from trainer import VecFrameStack

        class FakeVenv:
            num_agents = 1

            def step_async(self, actions):
                pass

            def step_wait(self):
                obs = {'matrix': np.full((3, 4, 4), 200, dtype=np.uint8), 'sectors': np.zeros(99, dtype=np.float32)}
                return [obs], [0.0], [False], [{'length': 0}]

        stack = VecFrameStack(FakeVenv(), k=4)
        stack._fill_frames(0, np.zeros((3, 4, 4), dtype=np.uint8))
        obs_list, _, _, _ = stack.step(None)
        self.assertEqual(obs_list[0]['matrix'].dtype, np.uint8)
        self.assertEqual(obs_list[0]['matrix'].shape, (12, 4, 4))
        self.assertEqual(int(obs_list[0]['matrix'][-1, 0, 0]), 200)


if __name__ == '__main__':
    unittest.main()
