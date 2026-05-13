#!/usr/bin/env bash
# stop_bot.sh — zatrzymuje trainer.py i karpathy_mod_runner.py wraz z podprocesami

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── helpers ────────────────────────────────────────────────────────────────────

killed=0

find_pids() {
    pgrep -f "$1" 2>/dev/null || true
}

stop_procs() {
    local label="$1"
    local pattern="$2"
    local pids
    pids=$(find_pids "$pattern")
    [[ -z "$pids" ]] && return

    echo -e "${YELLOW}[~]${NC} Stopping $label (PID: $(echo $pids | tr '\n' ' '))"
    kill -TERM $pids 2>/dev/null || true

    # Give 3s to exit cleanly, then SIGKILL
    local deadline=$(( $(date +%s) + 3 ))
    while [[ $(date +%s) -lt $deadline ]]; do
        local alive=0
        for pid in $pids; do
            kill -0 "$pid" 2>/dev/null && alive=1 || true
        done
        [[ $alive -eq 0 ]] && break
        sleep 0.5
    done

    for pid in $pids; do
        kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
    done

    echo -e "${GREEN}[OK]${NC} $label"
    killed=1
}

show_status() {
    echo -e "${BLUE}=== Status ===${NC}"
    local found=0

    for pattern in "karpathy_mod_runner.py" "trainer\.py" "chromedriver" "auto_report\.sh"; do
        local pids
        pids=$(find_pids "$pattern")
        if [[ -n "$pids" ]]; then
            echo -e "  ${YELLOW}RUNNING${NC}  $pattern  (PID: $(echo $pids | tr '\n' ' '))"
            found=1
        fi
    done

    # Chrome launched by selenium (has --remote-debugging-port)
    local chrome_pids
    chrome_pids=$(find_pids -- "--remote-debugging-port")
    if [[ -n "$chrome_pids" ]]; then
        echo -e "  ${YELLOW}RUNNING${NC}  selenium chrome  (PID: $(echo $chrome_pids | tr '\n' ' '))"
        found=1
    fi

    [[ $found -eq 0 ]] && echo "  Nothing running."
}

# ── main ───────────────────────────────────────────────────────────────────────

case "${1:-stop}" in
    status|--status|-s)
        show_status
        exit 0
        ;;
    stop|--stop|"")
        ;;
    *)
        echo "Usage: $0 [stop|status]"
        exit 1
        ;;
esac

echo -e "${BLUE}=== SlitherBot Stop ===${NC}"

# 1. Kill karpathy runner first — tak żeby nie restartowal workerów
stop_procs "karpathy runner"  "karpathy_mod_runner.py"

# 2. Kill trainer instances (normal + karpathy workers mają KARPATHY_EXPERIMENT_ID w env)
stop_procs "trainer(s)"       "python.*trainer\.py"

# 3. Kill chromedriver (selenium)
stop_procs "chromedriver"     "chromedriver"

# 4. Kill Chrome uruchomiony przez selenium — ma --remote-debugging-port w cmdline
#    NIE dotyka normalnego Chrome użytkownika
stop_procs "selenium chrome"  -- "--remote-debugging-port"

# 5. Posprzątaj worktree jeśli karpathy zostawił bałagan
if [[ -d "$SCRIPT_DIR/.karpathy_worktrees" ]]; then
    echo -e "${YELLOW}[~]${NC} Cleaning up karpathy worktrees..."
    python3 "$SCRIPT_DIR/karpathy_mod_runner.py" --cleanup 2>/dev/null || \
        rm -rf "$SCRIPT_DIR/.karpathy_worktrees"
    echo -e "${GREEN}[OK]${NC} Worktrees removed"
fi

if [[ $killed -eq 0 ]]; then
    echo "Nothing was running."
else
    echo -e "\n${GREEN}Done.${NC}"
fi
