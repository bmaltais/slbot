#!/usr/bin/env bash
# reset_training.sh — wipe all run artifacts so the next trainer.py starts from scratch.
#
# Removes: logs/, events/, training_stats.csv (+ recent/temp variants), nohup.out,
#          progress_report.md, checkpoint.pth, backup_models/*.pth, Python caches.
# Keeps:   source code, charts/, karpathy_mod_* experiment files.
#
# Usage:
#   ./reset_training.sh            # ask before touching model weights
#   ./reset_training.sh --yes      # no prompts
#   ./reset_training.sh --keep-models   # wipe logs/stats only, leave checkpoint + backups
#   ./reset_training.sh --dry-run  # show what would be removed
#
# Model weights are moved to ./old_runs/<timestamp>/ instead of deleted
# (old_runs/ is gitignored) unless --no-archive.

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ASSUME_YES=0
KEEP_MODELS=0
DRY_RUN=0
ARCHIVE=1

for arg in "$@"; do
    case "$arg" in
        -y|--yes)        ASSUME_YES=1 ;;
        --keep-models)   KEEP_MODELS=1 ;;
        -n|--dry-run)    DRY_RUN=1 ;;
        --no-archive)    ARCHIVE=0 ;;
        -h|--help)       sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo -e "${RED}[!]${NC} Unknown option: $arg"; exit 1 ;;
    esac
done

# ── safety: refuse to run while training is active ────────────────────────────
if pgrep -f "trainer.py|worker_process.py|karpathy_mod_runner.py" >/dev/null 2>&1; then
    echo -e "${RED}[!]${NC} Training processes are running. Stop them first: ./stop_bot.sh"
    exit 1
fi

run() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo -e "${YELLOW}[dry]${NC} $*"
    else
        "$@"
    fi
}

remove() {
    # remove <label> <path...>; skips paths that don't exist
    local label="$1"; shift
    local existing=()
    for p in "$@"; do
        [[ -e "$p" ]] && existing+=("$p")
    done
    [[ ${#existing[@]} -eq 0 ]] && return 0
    echo -e "${GREEN}[-]${NC} $label: ${existing[*]}"
    run rm -rf -- "${existing[@]}"
}

# ── logs, events, stats ───────────────────────────────────────────────────────
shopt -s nullglob
remove "logs"          logs/*
remove "death events"  events/*.json
remove "stats"         training_stats.csv recent_stats.csv temp_stats.csv
remove "misc output"   nohup.out progress_report.md fatal_error.log chromedriver.log geckodriver.log
remove "py caches"     __pycache__ tests/__pycache__ .pytest_cache

# ── model weights ─────────────────────────────────────────────────────────────
if [[ $KEEP_MODELS -eq 0 ]]; then
    models=(checkpoint.pth backup_models/*.pth)
    existing=()
    for p in "${models[@]}"; do [[ -e "$p" ]] && existing+=("$p"); done

    if [[ ${#existing[@]} -gt 0 ]]; then
        if [[ $ASSUME_YES -eq 0 && $DRY_RUN -eq 0 ]]; then
            echo -e "${YELLOW}[?]${NC} Found trained weights: ${existing[*]}"
            read -r -p "    Remove them (fresh model on next run)? [y/N] " ans
            [[ "${ans,,}" == "y" ]] || { echo -e "${YELLOW}[~]${NC} Keeping model weights."; exit 0; }
        fi
        if [[ $ARCHIVE -eq 1 ]]; then
            dest="old_runs/$(date +%Y%m%d_%H%M%S)"
            echo -e "${GREEN}[>]${NC} Archiving weights to $dest/"
            run mkdir -p "$dest"
            for p in "${existing[@]}"; do run mv -- "$p" "$dest/"; done
        else
            remove "model weights" "${existing[@]}"
        fi
    fi
fi

echo -e "${GREEN}[✓]${NC} Reset complete. Next 'python trainer.py' starts a fresh run."
