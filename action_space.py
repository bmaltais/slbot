"""The bot's discrete action space, shared by the env, the agent and demos.

Every module that needs to know what action N *means* imports from here so
the table has one home. ``slither_env.step`` decodes an action into a heading
change and a boost flag; ``label_human_action`` does the inverse for
recorded human play, snapping a continuous mouse heading onto the nearest
action so demonstrations can be replayed through the same network head.

The turn magnitudes are radians relative to the current heading; the human
labeller picks the nearest one, so the bucket edges are the midpoints.
"""
import math

ACTION_DIM = 14
ACTION_STRAIGHT = 0
ACTION_BOOST = 11
ACTION_BOOST_LEFT = 12
ACTION_BOOST_RIGHT = 13

# Heading change (radians) per action. Negative = left, positive = right.
ACTION_ANGLE_CHANGE = (
    0.0,     # 0  keep current direction
    -0.18,   # 1  left micro   (~10 deg)
    0.18,    # 2  right micro  (~10 deg)
    -0.35,   # 3  left gentle  (~20 deg)
    0.35,    # 4  right gentle (~20 deg)
    -0.61,   # 5  left medium  (~35 deg)
    0.61,    # 6  right medium (~35 deg)
    -0.96,   # 7  left sharp   (~55 deg)
    0.96,    # 8  right sharp  (~55 deg)
    -1.57,   # 9  left u-turn  (~90 deg)
    1.57,    # 10 right u-turn (~90 deg)
    0.0,     # 11 boost straight
    -0.18,   # 12 boost + left micro
    0.18,    # 13 boost + right micro
)

ACTION_IS_BOOST = tuple(a >= ACTION_BOOST for a in range(ACTION_DIM))

ACTION_NAMES = (
    'FWD', 'ML', 'MR', 'L1', 'R1', 'L2', 'R2', 'L3', 'R3', 'LU', 'RU',
    'BST', 'BL', 'BR',
)

_TURN_ACTIONS = tuple(range(0, ACTION_BOOST))
_BOOST_ACTIONS = (ACTION_BOOST, ACTION_BOOST_LEFT, ACTION_BOOST_RIGHT)
# A boosting human turning harder than this is labelled with the turn alone:
# the action space cannot express boost + sharp turn, and the turn is the
# part that keeps the snake alive. Midpoint between micro (0.18) and gentle
# (0.35), i.e. the edge of the boost+micro bucket.
BOOST_TURN_LIMIT = 0.265


def decode_action(action):
    """action id -> (angle_change radians, boost 0/1)."""
    a = int(action)
    if not 0 <= a < ACTION_DIM:
        raise ValueError(f"action {action} outside [0, {ACTION_DIM})")
    return ACTION_ANGLE_CHANGE[a], (1 if ACTION_IS_BOOST[a] else 0)


def wrap_angle(a):
    """Wrap to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _nearest(delta, candidates):
    return min(candidates, key=lambda a: abs(ACTION_ANGLE_CHANGE[a] - delta))


def label_human_action(heading, wanted_heading, boosting):
    """Snap a human's steering onto the nearest discrete action.

    heading: the snake's current heading ``ang`` (radians).
    wanted_heading: the mouse-driven target heading ``wang``; None if the
        game did not report one (labelled as keep-direction).
    boosting: whether the snake was boosting during the step.

    Deltas beyond a u-turn clamp to the u-turn bucket. A boosting turn up to
    BOOST_TURN_LIMIT keeps the boost (boost+micro); a harder boosting turn
    drops the boost in favour of the turn.
    """
    if wanted_heading is None or heading is None:
        delta = 0.0
    else:
        delta = wrap_angle(float(wanted_heading) - float(heading))
    if boosting and abs(delta) <= BOOST_TURN_LIMIT:
        return _nearest(delta, _BOOST_ACTIONS)
    return _nearest(delta, _TURN_ACTIONS)
