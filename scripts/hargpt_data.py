from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .video_ltc import ensure_video_timecode_csv, load_record_date_from_metadata_json

from .hargpt_pipeline_config import PipelineConfig


@dataclass
class PreparedData:
    config: PipelineConfig
    video_file: Path
    video_timecode_file: Path
    video_metadata_file: Path
    record_date: object
    video_start: pd.Timestamp
    video_end: pd.Timestamp
    raw_acc: pd.DataFrame
    acc_resampled: pd.DataFrame
    acc_video: pd.DataFrame
    window_index_df: pd.DataFrame


def parse_time_column(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str).str.strip(), errors="coerce")


def load_cup_acc_csv(acc_file: Path) -> pd.DataFrame:
    df = pd.read_csv(acc_file)
    df = df.rename(columns={"X": "x", "Y": "y", "Z": "z"})
    df = df[["time", "x", "y", "z"]].copy()
    df["time"] = parse_time_column(df["time"])
    df = df.dropna(subset=["time", "x", "y", "z"]).sort_values("time")
    df = df.drop_duplicates(subset=["time"]).reset_index(drop=True)
    return df


def resample_acc(df: pd.DataFrame, target_fs: float) -> pd.DataFrame:
    freq_ms = int(round(1000.0 / target_fs))
    rule = f"{freq_ms}ms"
    return (
        df.set_index("time")[["x", "y", "z"]]
        .resample(rule)
        .mean()
        .interpolate(method="time")
        .dropna()
        .reset_index()
    )


def load_video_range(video_timecode_file: Path, record_date) -> tuple[pd.Timestamp, pd.Timestamp]:
    vid = pd.read_csv(video_timecode_file)
    vid["tc_hms"] = vid["tc_str"].astype(str).str.slice(0, 8)
    vid["video_datetime"] = pd.to_datetime(
        record_date.strftime("%Y-%m-%d") + " " + vid["tc_hms"],
        errors="coerce",
    )
    vid = vid.dropna(subset=["video_datetime"]).sort_values("video_datetime").reset_index(drop=True)
    return vid["video_datetime"].iloc[0], vid["video_datetime"].iloc[-1]


def split_windows(df: pd.DataFrame, window_size: int, stride_size: int) -> pd.DataFrame:
    rows = []
    if len(df) < window_size:
        return pd.DataFrame(rows)
    for start in range(0, len(df) - window_size + 1, stride_size):
        end = start + window_size
        window_df = df.iloc[start:end].copy().reset_index(drop=True)
        rows.append(
            {
                "window_id": len(rows),
                "start_idx_in_video_subset": int(start),
                "end_idx_in_video_subset": int(end - 1),
                "start_time": window_df.loc[0, "time"],
                "end_time": window_df.loc[len(window_df) - 1, "time"],
            }
        )
    return pd.DataFrame(rows)


def prepare_video_aligned_data(config: PipelineConfig) -> PreparedData:
    video_file, video_timecode_file, video_metadata_file = ensure_video_timecode_csv(
        video_name=config.video_name,
        video_file=config.video_file,
        videos_dir=config.video_dir,
        fallback_timecode=config.video_start_timecode,
        fallback_fps=config.video_fps,
        regenerate=config.force_regenerate_video_metadata,
    )
    record_date = load_record_date_from_metadata_json(video_metadata_file)

    raw_acc = load_cup_acc_csv(config.acc_file)
    acc_resampled = resample_acc(raw_acc, config.target_fs)
    video_start, video_end = load_video_range(video_timecode_file, record_date)
    acc_video = acc_resampled[
        (acc_resampled["time"] >= video_start) & (acc_resampled["time"] <= video_end)
    ].copy().reset_index(drop=True)
    if len(acc_video) < config.window_size:
        raise ValueError("Video subset is shorter than one analysis window.")

    acc_video.to_csv(config.acc_video_csv, index=False)
    window_index_df = split_windows(acc_video, config.window_size, config.stride_size)
    window_index_df.to_csv(config.window_index_csv, index=False)

    return PreparedData(
        config=config,
        video_file=video_file,
        video_timecode_file=video_timecode_file,
        video_metadata_file=video_metadata_file,
        record_date=record_date,
        video_start=video_start,
        video_end=video_end,
        raw_acc=raw_acc,
        acc_resampled=acc_resampled,
        acc_video=acc_video,
        window_index_df=window_index_df,
    )


def summarize_prepared_data(prepared: PreparedData) -> None:
    print(f"Video file: {prepared.video_file}")
    print(f"Video timecode CSV: {prepared.video_timecode_file}")
    print(f"Video metadata JSON: {prepared.video_metadata_file}")
    print(f"ACC video subset saved to: {prepared.config.acc_video_csv}")
    print(f"Window index saved to: {prepared.config.window_index_csv}")
    print(f"Video range: {prepared.video_start} -> {prepared.video_end}")
    print(f"Video-subset windows: {len(prepared.window_index_df)}")
