from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .hargpt_data import load_cup_acc_csv, resample_acc, split_windows
from .hargpt_labels import LABEL_PRIORITY, LABEL_TO_STATE, STATE_COLORS, STATE_TO_NAME
from .hargpt_pipeline_config import PARTICIPANT_TO_ACCID
from .video_ltc import find_video_file, probe_video_metadata


@dataclass(frozen=True)
class CovfeeAnnotation:
    participant_no: int
    video_name: str
    label_file: Path

    @property
    def annotation_id(self) -> str:
        return f"{self.participant_no:02d}_{self.video_name}"

    @property
    def participant_id(self) -> str:
        return str(PARTICIPANT_TO_ACCID[self.participant_no])


@dataclass(frozen=True)
class CovfeeVideoRange:
    video_file: Path
    source_video_start: pd.Timestamp
    clip_start_offset_seconds: float
    clip_end_offset_seconds: float
    video_start: pd.Timestamp
    video_end: pd.Timestamp
    duration_seconds: float
    alignment_mode: str = "hardcoded_source_clip_offset"


COVFEE_VIDEO_CONFIG = {
    "v1": {
        "source_video_start": pd.Timestamp("2025-07-17 13:57:00")
        + pd.to_timedelta(10 / (60000 / 1001), unit="s"),
        "clip_start_offset_seconds": 5 * 60 + 30,
        "clip_end_offset_seconds": 10 * 60 + 30,
    },
    "v2": {
        "source_video_start": pd.Timestamp("2025-07-17 15:17:32")
        + pd.to_timedelta(7 / (60000 / 1001), unit="s"),
        "clip_start_offset_seconds": 3 * 60,
        "clip_end_offset_seconds": 8 * 60,
    },
}

ANNOTATION_STEM_PARTS = 2
TARGET_FS = 50.0
WINDOW_SECONDS = 2.0
STRIDE_SECONDS = 2.0
UNCERTAIN_LABELS = {"uncertain", "unknown", "ambiguous"}


def load_cup_acc_folder(acc_dir: Path) -> pd.DataFrame:
    acc_files = sorted(Path(acc_dir).glob("ACC_*.csv"))
    if not acc_files:
        raise FileNotFoundError(f"No ACC_*.csv files found in {acc_dir}")
    frames = []
    for acc_file in acc_files:
        frame = load_cup_acc_csv(acc_file)
        frame["source_acc_file"] = acc_file.name
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["time", "x", "y", "z"])
    combined = combined.sort_values("time").reset_index(drop=True)
    return combined


def discover_covfee_annotations(annotation_dir: Path) -> list[CovfeeAnnotation]:
    annotations = []
    for label_file in sorted(Path(annotation_dir).glob("*.json")):
        stem_parts = label_file.stem.lower().split("_")
        if len(stem_parts) != ANNOTATION_STEM_PARTS:
            continue
        participant_text, video_name = stem_parts
        if not participant_text.isdigit():
            continue
        participant_no = int(participant_text)
        if participant_no not in PARTICIPANT_TO_ACCID:
            raise ValueError(f"No ACC mapping found for COVFEE participant {participant_no:02d}.")
        if video_name not in COVFEE_VIDEO_CONFIG:
            raise ValueError(f"Unexpected COVFEE video name: {video_name!r}.")
        annotations.append(
            CovfeeAnnotation(
                participant_no=participant_no,
                video_name=video_name,
                label_file=label_file,
            )
        )
    return annotations


def _load_video_duration(project_root: Path, video_name: str) -> tuple[Path, float]:
    video_file = find_video_file(video_name, project_root / "videos")
    metadata = probe_video_metadata(video_file)
    duration_seconds = float(metadata.get("format", {}).get("duration", 0.0))
    if duration_seconds <= 0:
        raise ValueError(f"Could not resolve duration for video: {video_file}")
    return video_file, duration_seconds


def _build_acc_cache(
    project_root: Path,
    annotations: list[CovfeeAnnotation],
    target_fs: float,
) -> dict[str, dict[str, object]]:
    acc_cache = {}
    for annotation in annotations:
        if annotation.participant_id in acc_cache:
            continue
        acc_dir = project_root / "acc_files_cup" / annotation.participant_id
        raw_acc = load_cup_acc_folder(acc_dir)
        acc_resampled = resample_acc(raw_acc, target_fs)
        acc_cache[annotation.participant_id] = {
            "acc_dir": acc_dir,
            "acc_files": ";".join(sorted(raw_acc["source_acc_file"].unique())),
            "raw_acc": raw_acc,
            "acc_resampled": acc_resampled,
            "acc_start": acc_resampled["time"].iloc[0],
            "acc_end": acc_resampled["time"].iloc[-1],
        }
    return acc_cache


def _resolve_video_relative_ranges(
    project_root: Path,
    annotations: list[CovfeeAnnotation],
) -> dict[str, CovfeeVideoRange]:
    video_ranges = {}
    for video_name in sorted({annotation.video_name for annotation in annotations}):
        video_file, duration_seconds = _load_video_duration(project_root, video_name)
        video_config = COVFEE_VIDEO_CONFIG[video_name]
        source_video_start = video_config["source_video_start"]
        clip_start_offset_seconds = float(video_config["clip_start_offset_seconds"])
        clip_end_offset_seconds = float(video_config["clip_end_offset_seconds"])
        video_start = source_video_start + pd.to_timedelta(clip_start_offset_seconds, unit="s")
        video_end = source_video_start + pd.to_timedelta(clip_end_offset_seconds, unit="s")
        video_ranges[video_name] = CovfeeVideoRange(
            video_file=video_file,
            source_video_start=source_video_start,
            clip_start_offset_seconds=clip_start_offset_seconds,
            clip_end_offset_seconds=clip_end_offset_seconds,
            video_start=video_start,
            video_end=video_end,
            duration_seconds=min(duration_seconds, clip_end_offset_seconds - clip_start_offset_seconds),
        )
    return video_ranges


def _normalize_segments(label_file: Path) -> pd.DataFrame:
    segments = json.loads(label_file.read_text(encoding="utf-8"))
    label_df = pd.DataFrame(segments)
    required_cols = {"start", "end", "category"}
    missing_cols = required_cols - set(label_df.columns)
    if missing_cols:
        raise ValueError(f"{label_file} missing required fields: {sorted(missing_cols)}")

    label_df = label_df.copy()
    label_df["start"] = pd.to_numeric(label_df["start"], errors="coerce")
    label_df["end"] = pd.to_numeric(label_df["end"], errors="coerce")
    label_df["category"] = label_df["category"].astype(str).str.strip()
    label_df = label_df.dropna(subset=["start", "end", "category"]).sort_values(["start", "end"])
    label_df = label_df[label_df["end"] >= label_df["start"]].reset_index(drop=True)
    label_df = label_df[~label_df["category"].str.lower().isin(UNCERTAIN_LABELS)].reset_index(drop=True)

    unknown_categories = sorted(set(label_df["category"]) - set(LABEL_TO_STATE))
    if unknown_categories:
        raise ValueError(f"Unknown categories in {label_file}: {unknown_categories}")
    return label_df


def _label_acc_samples(acc_video: pd.DataFrame, segments: pd.DataFrame, video_start: pd.Timestamp) -> pd.DataFrame:
    labeled = acc_video.copy()
    labeled["time"] = pd.to_datetime(labeled["time"])
    labeled["state"] = LABEL_TO_STATE["Still"]
    labeled["manual_label"] = "Still"
    priority_buffer = np.zeros(len(labeled), dtype=int)

    for row in segments.itertuples(index=False):
        priority = LABEL_PRIORITY[row.category]
        abs_start = video_start + pd.to_timedelta(row.start, unit="s")
        abs_end = video_start + pd.to_timedelta(row.end, unit="s")
        mask = (labeled["time"] >= abs_start) & (labeled["time"] <= abs_end)
        if not mask.any():
            continue
        mask_np = mask.to_numpy()
        update_mask = mask_np & (priority >= priority_buffer)
        if not update_mask.any():
            continue
        priority_buffer[update_mask] = priority
        labeled.loc[update_mask, "state"] = LABEL_TO_STATE[row.category]
        labeled.loc[update_mask, "manual_label"] = row.category

    return labeled


def _plot_overlay(output_png: Path, labeled: pd.DataFrame, annotation: CovfeeAnnotation, video_start: pd.Timestamp) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plot_df = labeled.copy().sort_values("time").reset_index(drop=True)
    plot_df["time_rel_s"] = (plot_df["time"] - video_start).dt.total_seconds()

    runs = []
    if not plot_df.empty:
        start_idx = 0
        for i in range(1, len(plot_df) + 1):
            if i == len(plot_df) or plot_df.loc[i, "manual_label"] != plot_df.loc[i - 1, "manual_label"]:
                runs.append(
                    (
                        plot_df.loc[start_idx, "time_rel_s"],
                        plot_df.loc[i - 1, "time_rel_s"],
                        plot_df.loc[i - 1, "manual_label"],
                    )
                )
                start_idx = i

    fig, ax = plt.subplots(1, 1, figsize=(18, 5))
    for start_time, end_time, label in runs:
        state = LABEL_TO_STATE.get(str(label), 0)
        ax.axvspan(start_time, end_time, color=STATE_COLORS[state], alpha=0.28)
    ax.plot(plot_df["time_rel_s"], plot_df["x"], label="x", color="blue", linewidth=0.7)
    ax.plot(plot_df["time_rel_s"], plot_df["y"], label="y", color="red", linewidth=0.7)
    ax.plot(plot_df["time_rel_s"], plot_df["z"], label="z", color="black", linewidth=0.7)
    ax.set_title(f"COVFEE {annotation.annotation_id} ACC with manual-label overlay")
    ax.set_xlabel("Video relative time (s)")
    ax.set_ylabel("Acceleration")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def preprocess_covfee_annotations(
    project_root: Path,
    *,
    output_dir: Path | None = None,
    target_fs: float = TARGET_FS,
    window_seconds: float = WINDOW_SECONDS,
    stride_seconds: float = STRIDE_SECONDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir) if output_dir is not None else project_root / "outputs" / "covfee"
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = discover_covfee_annotations(project_root / "weak_label" / "covfee anno")
    all_labeled = []
    summary_rows = []

    acc_cache = _build_acc_cache(project_root, annotations, target_fs)
    video_ranges = _resolve_video_relative_ranges(project_root, annotations)

    for annotation in annotations:
        video_range = video_ranges[annotation.video_name]
        video_start = video_range.video_start
        video_end = video_range.video_end
        video_file = video_range.video_file
        cache_entry = acc_cache[annotation.participant_id]
        acc_dir = cache_entry["acc_dir"]
        segments = _normalize_segments(annotation.label_file)
        acc_resampled = cache_entry["acc_resampled"]
        acc_video = acc_resampled[(acc_resampled["time"] >= video_start) & (acc_resampled["time"] <= video_end)].copy()
        alignment_mode = video_range.alignment_mode

        acc_video = acc_video.reset_index(drop=True)
        if acc_video.empty:
            summary_rows.append(
                {
                    "annotation_id": annotation.annotation_id,
                    "participant_no": annotation.participant_no,
                    "participant_id": annotation.participant_id,
                    "video_name": annotation.video_name,
                    "status": "skipped_no_acc_overlap",
                    "alignment_mode": alignment_mode,
                    "rows": 0,
                    "windows": 0,
                    "video_start": video_start,
                    "video_end": video_end,
                    "source_video_start": video_range.source_video_start,
                    "clip_start_offset_seconds": video_range.clip_start_offset_seconds,
                    "clip_end_offset_seconds": video_range.clip_end_offset_seconds,
                    "video_duration_seconds": video_range.duration_seconds,
                    "acc_start": cache_entry["acc_start"],
                    "acc_end": cache_entry["acc_end"],
                    "overlap_seconds": 0.0,
                    "label_file": str(annotation.label_file),
                    "acc_dir": str(acc_dir),
                    "acc_files": cache_entry["acc_files"],
                    "video_file": str(video_file),
                }
            )
            continue

        labeled = _label_acc_samples(acc_video, segments, video_start)
        labeled["participant_no"] = annotation.participant_no
        labeled["participant_id"] = annotation.participant_id
        labeled["video_name"] = annotation.video_name
        labeled["annotation_id"] = annotation.annotation_id
        labeled["alignment_mode"] = alignment_mode
        labeled = labeled[
            [
                "time",
                "x",
                "y",
                "z",
                "state",
                "manual_label",
                "video_name",
                "participant_no",
                "participant_id",
                "annotation_id",
                "alignment_mode",
            ]
        ]

        annotation_dir = output_dir / annotation.annotation_id
        annotation_dir.mkdir(parents=True, exist_ok=True)
        labeled_csv = annotation_dir / f"manual_labels_{annotation.annotation_id}.csv"
        window_index_csv = annotation_dir / f"window_index_{annotation.annotation_id}.csv"
        overlay_png = annotation_dir / f"manual_labels_overlay_{annotation.annotation_id}.png"

        labeled.to_csv(labeled_csv, index=False)
        window_index_df = split_windows(
            labeled,
            int(round(target_fs * window_seconds)),
            int(round(target_fs * stride_seconds)),
        )
        window_index_df.to_csv(window_index_csv, index=False)
        _plot_overlay(overlay_png, labeled, annotation, video_start)
        all_labeled.append(labeled)

        summary_rows.append(
            {
                "annotation_id": annotation.annotation_id,
                "participant_no": annotation.participant_no,
                "participant_id": annotation.participant_id,
                "video_name": annotation.video_name,
                "status": "ok",
                "alignment_mode": alignment_mode,
                "rows": len(labeled),
                "windows": len(window_index_df),
                "video_start": video_start,
                "video_end": video_end,
                "source_video_start": video_range.source_video_start,
                "clip_start_offset_seconds": video_range.clip_start_offset_seconds,
                "clip_end_offset_seconds": video_range.clip_end_offset_seconds,
                "video_duration_seconds": video_range.duration_seconds,
                "acc_start": cache_entry["acc_start"],
                "acc_end": cache_entry["acc_end"],
                "overlap_seconds": (
                    min(cache_entry["acc_end"], video_end) - max(cache_entry["acc_start"], video_start)
                ).total_seconds(),
                "label_file": str(annotation.label_file),
                "acc_dir": str(acc_dir),
                "acc_files": cache_entry["acc_files"],
                "video_file": str(video_file),
                "labeled_csv": str(labeled_csv),
                "window_index_csv": str(window_index_csv),
                "overlay_png": str(overlay_png),
            }
        )

    combined = pd.concat(all_labeled, ignore_index=True) if all_labeled else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not combined.empty:
        combined["preprocessed_at"] = timestamp
        combined.to_csv(output_dir / "covfee_manual_labels_combined.csv", index=False)
    summary.to_csv(output_dir / "covfee_preprocessing_summary.csv", index=False)
    return combined, summary
