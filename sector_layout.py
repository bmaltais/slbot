"""Layout of the egocentric sector vector fed to the model's radar branch.

Kept dependency-free so config, workers, tests and the env can all import
it without pulling in torch or Selenium.

The first 99 floats are the alpha-5 layout and are left exactly where they
were so the emergency reflexes in agent.py keep their indices. Everything
after that was added so the network can separate *how much* food sits in a
direction from *how far* it is (a dead snake's remains 1800 units out vs a
crumb 100 units out used to score the same), tell walls from snake bodies,
and know how big the snake in each direction is.
"""

NUM_SECTORS = 24               # 15 degrees each, sector 0 = straight ahead, clockwise

# --- legacy alpha-5 block (indices frozen; agent._check_reflexes reads them) ---
FOOD_SCORE = 0                 # [0..23]   distance-weighted pellet mass, log-squashed
OBSTACLE_SCORE = 24            # [24..47]  closeness of nearest enemy/wall (1 = touching)
OBSTACLE_TYPE = 48             # [48..71]  -1 none, 0 body/wall, 1 enemy head
ENEMY_APPROACH = 72            # [72..95]  relative-velocity closing rate of the head, -1..1
G_WALL_DIST = 96               # dist_to_wall / scope
G_LENGTH = 97                  # own length / 500
G_SPEED = 98                   # own speed / 20
LEGACY_DIM = 99

# --- added bands ---
FOOD_MASS = 99                 # [99..122]  total pellet mass in sector (no distance weighting), log-squashed
FOOD_DIST = 123                # [123..146] 1 - mass-weighted mean pellet distance / scope; 0 = no food
WALL_DIST = 147                # [147..170] 1 - distance to wall along sector centre ray / scope; 0 = beyond scope
ENEMY_SIZE = 171               # [171..194] scale (sc / 6) of the closest enemy in the sector; 0 = none

# --- added globals ---
G_BOOST = 195                  # 1 when own speed says we are boosting
G_FOOD_TOTAL = 196             # total pellet mass within scope, log-squashed
G_TARGET_DIST = 197            # 1 - distance to the locked food target / scope (the compass target)
G_TARGET_MASS = 198            # mass of that target, log-squashed
G_TARGET_SIN = 199             # sin of the target's egocentric bearing (0 = ahead, clockwise)
G_TARGET_COS = 200             # cos of that bearing

SECTOR_DIM = 201


def sector_band(start):
    """slice covering the 24 floats of one per-sector band."""
    return slice(start, start + NUM_SECTORS)
