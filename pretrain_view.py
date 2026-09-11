"""Terminal view for the demonstration-pretraining phase.

Pretraining runs before the TUI dashboard exists and the trainer's logger
only writes to files, so this module turns DDQNAgent.pretrain_from_demos
progress updates into something to watch: a progress bar with ETA, the loss
and Q numbers, sparkline trends, and the one figure that says whether the
demos are taking hold, how often the greedy policy now picks the human's
action on sampled demo states, next to the human-vs-policy action mix.

Uses rich when it is installed and stdout is a terminal; otherwise prints
one plain line per update. Both paths share the same formatting helpers.
"""
import sys

import numpy as np

from action_space import ACTION_DIM, ACTION_NAMES

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without rich
    RICH_AVAILABLE = False

SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values, width=24):
    """Unicode sparkline of the last `width` values (flat line if constant)."""
    vals = [float(v) for v in list(values)[-width:] if v is not None and np.isfinite(v)]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return SPARK_CHARS[3] * len(vals)
    out = []
    for v in vals:
        idx = int((v - lo) / (hi - lo) * (len(SPARK_CHARS) - 1))
        out.append(SPARK_CHARS[idx])
    return "".join(out)


def fmt_duration(seconds):
    s = max(0, int(round(float(seconds))))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def action_mix(counts, top=5):
    """'FWD 80%  ML 4%  ...' for the `top` most frequent actions."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return "-"
    order = np.argsort(-counts)[:top]
    return "  ".join(f"{ACTION_NAMES[a]} {counts[a] / total:.0%}" for a in order if counts[a] > 0)


def progress_bar(fraction, width=28):
    fraction = min(1.0, max(0.0, float(fraction)))
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


def progress_line(update):
    """One-line plain-text summary of a pretraining update."""
    m = update['metrics']
    agree = update.get('agreement')
    parts = [
        f"[Pretrain] {update['step']}/{update['steps']} ({update['step'] / update['steps']:.0%})"
        + (f" epoch {update['epoch']}/{update['epochs']}" if update.get('epochs') else ""),
        f"loss {m['loss']:.4f}",
        f"margin {m['margin_loss']:.4f}",
        f"td {m['td_error_mean']:.3f}",
        f"q {m['q_mean']:.2f}/{m['q_max']:.2f}",
        f"grad {m['grad_norm']:.2f}",
    ]
    if agree:
        parts.append(f"agree {agree['agreement']:.0%}")
    parts.append(f"{fmt_duration(update['elapsed'])} elapsed, ~{fmt_duration(update['eta'])} left")
    return " | ".join(parts)


def demo_table_lines(summary):
    """Plain-text rows describing the loaded demo episodes."""
    lines = []
    for r in summary.get('episode_rows', []):
        end = "step cap" if r['truncated'] else (str(r['cause']) or "death")
        lines.append(
            f"  {r['file']:<44} {r['steps']:>5} steps  peak {r['peak_length']:>4}  "
            f"reward {r['total_reward']:>8.1f}  {end:<14} {action_mix(r['action_counts'], top=4)}"
        )
    for f in summary.get('skipped_files', []):
        lines.append(f"  {f:<44} skipped (peak length below --demo-min-score {summary.get('min_score', 0)})")
    return lines


class PretrainMonitor:
    """Context manager that renders pretraining updates.

    with PretrainMonitor() as mon:
        agent.pretrain_from_demos(steps, on_progress=mon.update)

    Keeps loss / margin / agreement histories for the trend lines. The last
    rendered panel stays on screen when the context exits.
    """

    def __init__(self, use_rich=None, console=None, stream=None):
        self.stream = stream or sys.stdout
        if use_rich is None:
            use_rich = RICH_AVAILABLE and hasattr(self.stream, 'isatty') and self.stream.isatty()
        self.use_rich = bool(use_rich and RICH_AVAILABLE)
        self.console = console or (Console(file=self.stream) if self.use_rich else None)
        self._live = None
        self.loss_hist = []
        self.margin_hist = []
        self.agree_hist = []
        self.last = None

    def __enter__(self):
        if self.use_rich:
            self._live = Live(Text("[Pretrain] starting..."), console=self.console,
                              refresh_per_second=4, transient=False)
            self._live.__enter__()
        return self

    def __exit__(self, *exc):
        if self._live is not None:
            self._live.__exit__(*exc)
            self._live = None
        return False

    def update(self, u):
        self.last = u
        self.loss_hist.append(u['metrics']['loss'])
        self.margin_hist.append(u['metrics']['margin_loss'])
        if u.get('agreement'):
            self.agree_hist.append(u['agreement']['agreement'])
        if self._live is not None:
            self._live.update(self.render())
        else:
            print(progress_line(u), file=self.stream, flush=True)

    def render(self):
        """Rich renderable for the latest update."""
        u = self.last
        m = u['metrics']
        frac = u['step'] / u['steps']
        head = Text.assemble(
            ("Pretraining on demonstrations  ", "bold"),
            (progress_bar(frac), "cyan"),
            f"  {u['step']}/{u['steps']} ({frac:.0%})",
            (f"   epoch {u['epoch']}/{u['epochs']}" if u.get('epochs') else ""),
        )
        timing = Text(
            f"{fmt_duration(u['elapsed'])} elapsed, about {fmt_duration(u['eta'])} left"
            "   (Ctrl+C stops early and saves)",
            style="dim",
        )

        t = Table.grid(padding=(0, 2))
        t.add_column(style="dim", justify="right")
        t.add_column()
        t.add_row("loss", f"{m['loss']:.4f}   {sparkline(self.loss_hist)}")
        t.add_row("margin loss", f"{m['margin_loss']:.4f}   {sparkline(self.margin_hist)}")
        t.add_row("td error", f"{m['td_error_mean']:.3f}      q mean {m['q_mean']:.2f}   q max {m['q_max']:.2f}   "
                              f"grad {m['grad_norm']:.2f}   lr {u['lr']:.1e}")

        agree = u.get('agreement')
        if agree:
            pct = agree['agreement']
            style = "green" if pct >= 0.6 else ("yellow" if pct >= 0.35 else "red")
            t.add_row("agrees with you",
                      Text.assemble((f"{pct:.0%}", f"bold {style}"),
                                    f" of {agree['n']} demo states   {sparkline(self.agree_hist)}"))
            t.add_row("your actions", action_mix(agree['human_counts']))
            t.add_row("policy picks", action_mix(agree['policy_counts']))

        note = Text(
            "margin loss -> 0 means your action is ranked on top; agreement rising means the network "
            "reproduces your choices, not just the reward.",
            style="dim",
        )
        return Panel(Group(head, timing, Text(""), t, Text(""), note), title="Demo pretraining", border_style="cyan")
