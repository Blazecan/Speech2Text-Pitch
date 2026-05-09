"""
Pitch Visualizer
Reads a pipeline log file (newline-delimited JSON, one WordPackage per line)
and renders a continuous pitch graph with word boundary boxes overlaid.

Usage:
    python pitch_visualizer.py pipeline_output.log
    python pitch_visualizer.py pipeline_output.log --min-confidence 0.4
    python pitch_visualizer.py pipeline_output.log --save pitch_graph.png
"""

import json
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import LogLocator, LogFormatter
from pathlib import Path


# ── Aesthetic constants ────────────────────────────────────────────────────────
BG_COLOR        = "#0d0f14"
PANEL_COLOR     = "#13161e"
GRID_COLOR      = "#1e2330"
PITCH_COLOR     = "#00e5ff"
PITCH_GLOW      = "#004d5e"
WORD_BOX_FACE   = "#1a2a1a"
WORD_BOX_EDGE   = "#39ff14"
WORD_TEXT_COLOR = "#39ff14"
AXIS_COLOR      = "#4a5568"
TICK_COLOR      = "#6b7280"
LABEL_COLOR     = "#9ca3af"
TITLE_COLOR     = "#e2e8f0"
UNVOICED_COLOR  = "#2a2f3e"


def load_log(path: str, min_confidence: float) -> list[dict]:
    """
    Parse the log file. Each line is a JSON WordPackage.
    Returns list of word dicts sorted by absolute_start.
    Filters out pitch frames below min_confidence.
    """
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                pkg = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed line {lineno}: {e}")
                continue

            # Filter pitch frames by confidence
            pkg["pitch_frames"] = [
                fr for fr in pkg.get("pitch_frames", [])
                if fr.get("confidence", 0) >= min_confidence
            ]
            words.append(pkg)

    words.sort(key=lambda w: w["absolute_start"])
    if words:
        t_offset = words[0]["absolute_start"]
        for word in words:
            word["absolute_start"] -= t_offset
            word["absolute_end"] -= t_offset
    return words


def build_pitch_series(words: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    times = []
    freqs = []

    for word in words:
        word_start = word["absolute_start"]
        word_end = word["absolute_end"]
        word_dur = word_end - word_start

        frames = word["pitch_frames"]
        voiced = [fr for fr in frames if fr.get("voiced", False) and fr["hz"] > 0]

        if not voiced:
            continue

        # Remap frame times to the word's absolute time window.
        # Frame times are relative to segment start, not word start, so we
        # normalize them to [0, 1] within the word's own frame range first,
        # then scale to the word's absolute duration.
        frame_times = [fr["time"] for fr in voiced]
        t_local_min = min(frame_times)
        t_local_max = max(frame_times)
        t_local_range = t_local_max - t_local_min if t_local_max > t_local_min else 1.0

        for fr in voiced:
            normalized = (fr["time"] - t_local_min) / t_local_range
            abs_t = word_start + normalized * word_dur
            times.append(abs_t)
            freqs.append(fr["hz"])

        times.append(np.nan)
        freqs.append(np.nan)

    return np.array(times, dtype=float), np.array(freqs, dtype=float)


def plot(words: list[dict], save_path: str | None):
    if not words:
        print("No word data found in log file.")
        sys.exit(1)

    times, freqs = build_pitch_series(words)

    # Overall time bounds
    t_min = words[0]["absolute_start"]
    t_max = words[-1]["absolute_end"]
    t_pad = (t_max - t_min) * 0.03
    x_min = t_min - t_pad
    x_max = t_max + t_pad

    # Pitch y-axis bounds (log scale — avoid zero)
    valid_freqs = freqs[~np.isnan(freqs)]
    if len(valid_freqs) == 0:
        print("No voiced pitch frames found. Check --min-confidence or log contents.")
        sys.exit(1)

    y_min = max(40.0,  np.percentile(valid_freqs, 2)  * 0.85)
    y_max = min(800.0, np.percentile(valid_freqs, 98) * 1.15)

    # ── Figure setup ──────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":      "monospace",
        "axes.facecolor":   PANEL_COLOR,
        "figure.facecolor": BG_COLOR,
        "text.color":       TITLE_COLOR,
    })

    fig, ax = plt.subplots(figsize=(max(14, len(words) * 1.4), 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(PANEL_COLOR)

    # ── Grid ──────────────────────────────────────────────────────────────────
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=2, numticks=10))
    ax.yaxis.set_major_formatter(LogFormatter(base=2, labelOnlyBase=False))
    ax.grid(which="major", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.grid(which="minor", color=GRID_COLOR, linewidth=0.3, zorder=0, alpha=0.5)

    # ── Word boundary boxes ───────────────────────────────────────────────────
    box_height_log_ratio = 0.12   # Fraction of log y-range for box height
    log_y_min = np.log10(y_min)
    log_y_max = np.log10(y_max)
    box_bottom_log = log_y_min - (log_y_max - log_y_min) * 0.18
    box_top_log    = log_y_min - (log_y_max - log_y_min) * 0.04
    box_bottom = 10 ** box_bottom_log
    box_top    = 10 ** box_top_log
    box_h      = box_top - box_bottom

    for word in words:
        w_start = word["absolute_start"]
        w_end   = word["absolute_end"]
        w_dur   = w_end - w_start
        label   = word["word"]

        rect = mpatches.FancyBboxPatch(
            (w_start, box_bottom),
            w_dur, box_h,
            boxstyle="round,pad=0.0",
            linewidth=1.2,
            edgecolor=WORD_BOX_EDGE,
            facecolor=WORD_BOX_FACE,
            zorder=3,
            clip_on=False
        )
        ax.add_patch(rect)

        ax.text(
            w_start + w_dur / 2,
            box_bottom + box_h / 2,
            label,
            ha="center", va="center",
            fontsize=8.5,
            color=WORD_TEXT_COLOR,
            fontweight="bold",
            fontfamily="monospace",
            zorder=4,
            clip_on=False
        )

        # Thin vertical tick connecting box top to plot bottom
        ax.axvline(x=w_start, ymin=0, color=WORD_BOX_EDGE,
                   linewidth=0.5, alpha=0.3, zorder=1)
        ax.axvline(x=w_end,   ymin=0, color=WORD_BOX_EDGE,
                   linewidth=0.5, alpha=0.3, zorder=1)

    # ── Pitch glow (thick, faint, same line drawn behind for glow effect) ─────
    ax.plot(times, freqs,
            color=PITCH_GLOW, linewidth=6, alpha=0.35,
            solid_capstyle="round", zorder=4)

    # ── Pitch line ────────────────────────────────────────────────────────────
    ax.plot(times, freqs,
            color=PITCH_COLOR, linewidth=1.8, alpha=0.95,
            solid_capstyle="round", zorder=5)

    # ── Axes limits and labels ────────────────────────────────────────────────
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(box_bottom * 0.6, y_max)

    ax.tick_params(colors=TICK_COLOR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(AXIS_COLOR)

    ax.set_xlabel("Time (s)", color=LABEL_COLOR, fontsize=10, labelpad=8)
    ax.set_ylabel("Pitch (Hz)", color=LABEL_COLOR, fontsize=10, labelpad=8)
    ax.set_title("Pitch Contour", color=TITLE_COLOR,
                 fontsize=14, fontweight="bold", pad=16, fontfamily="monospace")

    # ── Subtle note count in corner ───────────────────────────────────────────
    ax.text(0.99, 0.97,
            f"{len(words)} words  ·  {len(valid_freqs)} voiced frames",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=7.5, color=TICK_COLOR, fontfamily="monospace")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=BG_COLOR)
        print(f"Saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize pitch from pipeline log.")
    parser.add_argument("log", help="Path to the pipeline .log file")
    parser.add_argument(
        "--min-confidence", type=float, default=0.45,
        help="Minimum pitch frame confidence to include (default: 0.45)"
    )
    parser.add_argument(
        "--save", default=None, metavar="PATH",
        help="Save graph to file instead of displaying (e.g. pitch.png)"
    )
    args = parser.parse_args()

    if not Path(args.log).exists():
        print(f"Error: log file not found: {args.log}")
        sys.exit(1)

    words = load_log(args.log, args.min_confidence)
    print(f"Loaded {len(words)} words from {args.log}")
    plot(words, args.save)


if __name__ == "__main__":
    main()
