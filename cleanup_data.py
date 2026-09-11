
import os
import re
import time
from datetime import datetime, timedelta

# Directories
BACKUP_DIR = "backup_models"
EVENTS_DIR = "events"
KEEP_BACKUPS_DEFAULT = 10
KEEP_EVENTS_DEFAULT = 40


def configured_keep_backups(default=KEEP_BACKUPS_DEFAULT):
    raw = os.environ.get("SLBOT_KEEP_BACKUPS", "").strip()
    if not raw:
        return int(default)
    try:
        return max(1, int(raw))
    except ValueError:
        return int(default)


def configured_keep_events(default=KEEP_EVENTS_DEFAULT):
    raw = os.environ.get("SLBOT_KEEP_EVENTS", "").strip()
    if not raw:
        return int(default)
    try:
        return max(1, int(raw))
    except ValueError:
        return int(default)


_BACKUP_SCORE = re.compile(r'_s(\d+)_f(\d+)(?:_pk(\d+))?')

# Fitness weights for the "new best model" gate. Peak length (body-part
# count) is the closest thing to the real game objective: it reflects mass
# actually gained and lost, whereas food count treats a tiny orb and a big
# pile the same and steps reward mere survival. Its raw scale is ~5x smaller
# than food and ~15x smaller than steps, so it gets the largest weight to
# land all three terms in the same ballpark. Changing these invalidates the
# saved best-fitness bar; start the next run with --reset-best.
FITNESS_W_PEAK_LEN = 15.0
FITNESS_W_STEPS = 1.0
FITNESS_W_FOOD = 5.0


def compute_fitness(peak_len, steps, food):
    """Composite fitness used to decide whether a model is a new best."""
    return (
        peak_len * FITNESS_W_PEAK_LEN
        + steps * FITNESS_W_STEPS
        + food * FITNESS_W_FOOD
    )


def backup_fitness_from_name(name):
    """Fitness recovered from a backup filename (_s<steps>_f<food>_pk<peak>)."""
    m = _BACKUP_SCORE.search(name)
    if not m:
        return None
    steps, food = int(m.group(1)), int(m.group(2))
    peak = int(m.group(3) or 0)
    return compute_fitness(peak, steps, food)


def find_best_backup(backup_dir=BACKUP_DIR):
    """Path of the highest-fitness backup, or None."""
    if not os.path.isdir(backup_dir):
        return None
    scored = []
    for f in os.listdir(backup_dir):
        if not f.endswith(".pth"):
            continue
        fit = backup_fitness_from_name(f)
        if fit is None:
            continue
        path = os.path.join(backup_dir, f)
        scored.append((fit, os.path.getmtime(path), path))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return scored[0][2]


def prune_backups(keep=None, backup_dir=BACKUP_DIR):
    """Keep the newest `keep` .pth files in backup_dir. Returns removed count."""
    if keep is None:
        keep = configured_keep_backups()
    if keep < 1 or not os.path.isdir(backup_dir):
        return 0
    files = [
        os.path.join(backup_dir, f)
        for f in os.listdir(backup_dir)
        if f.endswith(".pth")
    ]
    if len(files) <= keep:
        return 0
    files.sort(key=os.path.getmtime, reverse=True)
    removed = 0
    for path in files[keep:]:
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    return removed


def prune_events(keep=None, events_dir=EVENTS_DIR):
    """Keep the newest `keep` event JSON packets and matching PNGs."""
    if keep is None:
        keep = configured_keep_events()
    if keep < 1 or not os.path.isdir(events_dir):
        return 0
    jsons = [
        os.path.join(events_dir, f)
        for f in os.listdir(events_dir)
        if f.endswith(".json")
    ]
    jsons.sort(key=os.path.getmtime, reverse=True)
    removed = 0
    for path in jsons[keep:]:
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
        png = path[:-5] + ".png"
        if os.path.isfile(png):
            try:
                os.remove(png)
                removed += 1
            except OSError:
                pass
    # Orphan PNGs with no json
    keep_stems = {os.path.splitext(p)[0] for p in jsons[:keep]}
    for f in os.listdir(events_dir):
        if not f.endswith(".png"):
            continue
        stem = os.path.join(events_dir, os.path.splitext(f)[0])
        if stem not in keep_stems:
            try:
                os.remove(os.path.join(events_dir, f))
                removed += 1
            except OSError:
                pass
    return removed

def cleanup_checkpoints(keep_top=20):
    """Keep only models with high steps/food and the most recent ones."""
    if not os.path.exists(BACKUP_DIR):
        return
    
    files = [f for f in os.listdir(BACKUP_DIR) if f.endswith('.pth')]
    if not files:
        return

    # Regex to extract: ep (episode), s (steps), f (food)
    # Example: best_model_20260217-f6da4990_ep7321_s586_f179.pth
    pattern = re.compile(r'_ep(\d+)_s(\d+)_f(\d+)\.pth')
    
    model_data = []
    for f in files:
        full_path = os.path.join(BACKUP_DIR, f)
        m = pattern.search(f)
        if m:
            ep, steps, food = map(int, m.groups())
            # Scoring: steps are primary (survival), food is secondary
            score = steps * 10 + food 
            model_data.append({
                'name': f,
                'path': full_path,
                'ep': ep,
                'steps': steps,
                'food': food,
                'score': score,
                'mtime': os.path.getmtime(full_path)
            })
        else:
            # Files without proper pattern (keep them just in case if they are new)
            if time.time() - os.path.getmtime(full_path) > 86400: # older than 1 day
                # print(f"Removing unknown file: {f}")
                # os.remove(full_path)
                pass

    # 1. Keep TOP X by Score (Best performers)
    top_by_score = sorted(model_data, key=lambda x: x['score'], reverse=True)[:keep_top]
    
    # 2. Keep TOP X by Recency (Latest checkpoints)
    top_by_time = sorted(model_data, key=lambda x: x['mtime'], reverse=True)[:keep_top]
    
    # Combine sets of names to keep
    to_keep = {m['name'] for m in top_by_score} | {m['name'] for m in top_by_time}
    
    removed_count = 0
    for m in model_data:
        if m['name'] not in to_keep:
            try:
                os.remove(m['path'])
                removed_count += 1
            except Exception as e:
                print(f"Error removing {m['name']}: {e}")
                
    if removed_count > 0:
        print(f"[Cleanup] Removed {removed_count} old/low-performance checkpoints. Kept {len(to_keep)} best/recent models.")

def cleanup_events(days_to_keep=3):
    """Remove old event logs and screenshots."""
    if not os.path.exists(EVENTS_DIR):
        return
    
    files = os.listdir(EVENTS_DIR)
    cutoff = time.time() - (days_to_keep * 86400)
    
    removed_count = 0
    for f in files:
        full_path = os.path.join(EVENTS_DIR, f)
        if os.path.getmtime(full_path) < cutoff:
            try:
                os.remove(full_path)
                removed_count += 1
            except Exception as e:
                print(f"Error removing event {f}: {e}")
                
    if removed_count > 0:
        print(f"[Cleanup] Removed {removed_count} old event files (older than {days_to_keep} days).")

if __name__ == "__main__":
    print(f"--- Data Cleanup Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    cleanup_checkpoints(keep_top=configured_keep_backups())
    prune_backups()
    prune_events()
    print("--- Cleanup Finished ---")
