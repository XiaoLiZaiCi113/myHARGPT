from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from .hargpt_data import PreparedData


LABEL_TO_STATE = {"Still": 0, "Gesture": 1, "Drinking": 2, "Toasting": 3, "Nodding": 4}
LABEL_PRIORITY = {"Still": 0, "Nodding": 1, "Gesture": 2, "Toasting": 3, "Drinking": 4}
STATE_TO_NAME = {0: "Still", 1: "Gesture", 2: "Drinking", 3: "Toasting", 4: "Nodding"}
STATE_COLORS = {0: "white", 1: "#f4a261", 2: "#2a9d8f", 3: "#577590", 4: "#b56576"}
CATEGORY_ALIASES = {
    "still": "Still",
    "gesture": "Gesture",
    "drink": "Drinking",
    "drinking": "Drinking",
    "toast": "Toasting",
    "toasting": "Toasting",
    "nodding": "Nodding",
}


def build_weak_label_csv(prepared: PreparedData) -> pd.DataFrame:
    config = prepared.config
    if not config.weak_label_input.exists():
        raise FileNotFoundError(f"Weak label file not found: {config.weak_label_input}")
    if config.manual_label_output is None:
        raise ValueError("Manual label output path has not been initialized. Run prepare_video_aligned_data first.")

    weak_segments = json.loads(config.weak_label_input.read_text(encoding="utf-8"))
    weak_df = pd.DataFrame(weak_segments)
    required_cols = {"start", "end", "category"}
    missing_cols = required_cols - set(weak_df.columns)
    if missing_cols:
        raise ValueError(f"Weak label JSON missing required fields: {sorted(missing_cols)}")

    weak_df = weak_df.copy()
    weak_df["start"] = pd.to_numeric(weak_df["start"], errors="coerce")
    weak_df["end"] = pd.to_numeric(weak_df["end"], errors="coerce")
    weak_df["category"] = weak_df["category"].astype(str).str.strip()
    weak_df["category"] = weak_df["category"].map(
        lambda value: CATEGORY_ALIASES.get(str(value).strip().lower(), str(value).strip())
    )
    weak_df = weak_df.dropna(subset=["start", "end", "category"]).sort_values(["start", "end"]).reset_index(drop=True)
    if not weak_df[weak_df["end"] < weak_df["start"]].empty:
        raise ValueError("Weak label contains segments with end < start.")

    unknown_categories = sorted(set(weak_df["category"]) - set(LABEL_TO_STATE))
    if unknown_categories:
        raise ValueError(f"Unknown weak label categories: {unknown_categories}")

    weak_df["abs_start"] = prepared.video_start + pd.to_timedelta(weak_df["start"], unit="s")
    weak_df["abs_end"] = prepared.video_start + pd.to_timedelta(weak_df["end"], unit="s")
    weak_df["clipped_start"] = weak_df["abs_start"].clip(lower=prepared.video_start, upper=prepared.video_end)
    weak_df["clipped_end"] = weak_df["abs_end"].clip(lower=prepared.video_start, upper=prepared.video_end)
    weak_df = weak_df[weak_df["clipped_end"] >= weak_df["clipped_start"]].reset_index(drop=True)

    acc_weak = prepared.acc_video.copy()
    acc_weak["time"] = pd.to_datetime(acc_weak["time"])
    acc_weak["state"] = LABEL_TO_STATE["Still"]
    acc_weak["manual_label"] = "Still"
    priority_buffer = np.zeros(len(acc_weak), dtype=int)

    for row in weak_df.itertuples(index=False):
        priority = LABEL_PRIORITY[row.category]
        mask = (acc_weak["time"] >= row.clipped_start) & (acc_weak["time"] <= row.clipped_end)
        if not mask.any():
            continue
        mask_np = mask.to_numpy()
        update_mask = mask_np & (priority >= priority_buffer)
        if not update_mask.any():
            continue
        priority_buffer[update_mask] = priority
        acc_weak.loc[update_mask, "state"] = LABEL_TO_STATE[row.category]
        acc_weak.loc[update_mask, "manual_label"] = row.category

    acc_weak["video_name"] = config.video_name
    acc_weak = acc_weak[["time", "x", "y", "z", "state", "manual_label", "video_name"]]
    acc_weak.to_csv(config.manual_label_output, index=False)
    return acc_weak


def summarize_weak_labels(prepared: PreparedData, acc_weak: pd.DataFrame) -> None:
    print(f"Weak label input: {prepared.config.weak_label_input}")
    print(f"Manual label output: {prepared.config.manual_label_output}")
    print(f"Video range: {prepared.video_start} -> {prepared.video_end}")
    print(f"ACC rows labeled: {len(acc_weak)}")
    print(acc_weak["manual_label"].value_counts().sort_index().rename_axis("manual_label"))


def plot_weak_label_overlay(prepared: PreparedData, acc_weak: pd.DataFrame) -> Path:
    if prepared.config.manual_label_overlay_png is None:
        raise ValueError("Manual label overlay path has not been initialized. Run prepare_video_aligned_data first.")
    acc_video = prepared.acc_video.copy()
    acc_video["time_rel_s"] = (acc_video["time"] - prepared.video_start).dt.total_seconds()
    weak_df = acc_weak.copy().sort_values("time").reset_index(drop=True)
    weak_df["time_rel_s"] = (weak_df["time"] - prepared.video_start).dt.total_seconds()

    weak_runs = []
    if not weak_df.empty:
        start_idx = 0
        for i in range(1, len(weak_df) + 1):
            if i == len(weak_df) or weak_df.loc[i, "manual_label"] != weak_df.loc[i - 1, "manual_label"]:
                weak_runs.append((weak_df.loc[start_idx, "time_rel_s"], weak_df.loc[i - 1, "time_rel_s"], weak_df.loc[i - 1, "manual_label"]))
                start_idx = i

    plt.style.use("default")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.edgecolor"] = "white"
    plt.rcParams["savefig.transparent"] = False

    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
    for start_time, end_time, weak_label in weak_runs:
        weak_state = LABEL_TO_STATE.get(str(weak_label), 0)
        ax.axvspan(start_time, end_time, color=STATE_COLORS[int(weak_state)], alpha=0.28)
    ax.plot(acc_video["time_rel_s"], acc_video["x"], label="x", color="blue", linewidth=0.7)
    ax.plot(acc_video["time_rel_s"], acc_video["y"], label="y", color="red", linewidth=0.7)
    ax.plot(acc_video["time_rel_s"], acc_video["z"], label="z", color="black", linewidth=0.7)
    ax.set_title("ACC over full video with weak-label overlay")
    ax.set_ylabel("Acceleration")
    ax.set_xlabel("Video relative time (s)")

    legend_handles = [Patch(facecolor=STATE_COLORS[s], edgecolor="gray", label=STATE_TO_NAME[s]) for s in [0, 1, 2, 3, 4]]
    line_handles, line_labels = ax.get_legend_handles_labels()
    ax.legend(line_handles + legend_handles, line_labels + [STATE_TO_NAME[s] for s in [0, 1, 2, 3, 4]], loc="upper right", ncol=3)

    prepared.config.manual_label_overlay_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(prepared.config.manual_label_overlay_png, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)
    return prepared.config.manual_label_overlay_png
