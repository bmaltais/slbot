from cleanup_data import (
    FITNESS_W_FOOD,
    FITNESS_W_PEAK_LEN,
    FITNESS_W_STEPS,
    backup_fitness_from_name,
    compute_fitness,
    find_best_backup,
)


def test_peak_length_dominates_at_typical_scale():
    # Stage-3 averages from a real run: peak≈28, steps≈475, food≈135.
    peak, steps, food = 28, 475, 135
    total = compute_fitness(peak, steps, food)
    assert peak * FITNESS_W_PEAK_LEN / total > 0.25
    assert food * FITNESS_W_FOOD / total < 0.5
    assert steps * FITNESS_W_STEPS / total < 0.5


def test_backup_name_matches_live_formula():
    name = "best_model_abc_ep10_s475_f135_pk28.pth"
    assert backup_fitness_from_name(name) == compute_fitness(28, 475, 135)


def test_backup_name_without_peak_and_garbage():
    assert backup_fitness_from_name("best_model_x_s100_f20.pth") == compute_fitness(0, 100, 20)
    assert backup_fitness_from_name("checkpoint.pth") is None


def test_find_best_backup_prefers_larger_snake(tmp_path):
    # Same steps and food; the one that grew bigger must win.
    small = tmp_path / "best_model_a_ep1_s400_f100_pk20.pth"
    big = tmp_path / "best_model_b_ep2_s400_f100_pk40.pth"
    small.write_bytes(b"x")
    big.write_bytes(b"x")
    assert find_best_backup(str(tmp_path)) == str(big)
