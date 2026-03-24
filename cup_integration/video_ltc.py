from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".mxf",
    ".avi",
    ".mkv",
    ".lrv",
    ".mpg",
    ".mpeg",
)


def run_command(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def parse_fraction(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text == "0/0":
        return None
    if "/" in text:
        numerator, denominator = map(float, text.split("/", 1))
        return numerator / denominator if denominator else None
    return float(text)


def sec_to_tc(seconds: float, fps: float) -> str:
    hours = int(seconds // 3600)
    seconds -= hours * 3600
    minutes = int(seconds // 60)
    seconds -= minutes * 60
    whole_seconds = int(seconds)
    seconds -= whole_seconds
    frames = int(round(seconds * fps))
    max_frames = max(1, int(round(fps)))
    if frames >= max_frames:
        frames = 0
        whole_seconds += 1
    if whole_seconds >= 60:
        whole_seconds = 0
        minutes += 1
    if minutes >= 60:
        minutes = 0
        hours += 1
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}:{frames:02d}"


def find_video_file(video_name: str, videos_dir: Path) -> Path:
    candidates = []
    for ext in VIDEO_EXTENSIONS:
        candidates.extend(sorted(videos_dir.glob(f"{video_name}{ext}")))
        candidates.extend(sorted(videos_dir.glob(f"{video_name}{ext.upper()}")))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a video named '{video_name}' in {videos_dir}. "
            f"Supported extensions: {', '.join(VIDEO_EXTENSIONS)}"
        )
    return candidates[0]


def probe_video_metadata(video_file: Path) -> dict[str, Any]:
    return json.loads(
        run_command(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(video_file),
            ]
        )
    )


def resolve_timecode_and_fps(
    video_file: Path,
    fallback_timecode: str | None = None,
    fallback_fps: float | None = None,
) -> tuple[str, float, dict[str, Any]]:
    meta = probe_video_metadata(video_file)
    streams = meta.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)

    timecode = None
    fps = None
    if video_stream is not None:
        tags = video_stream.get("tags", {})
        timecode = tags.get("timecode")
        fps = parse_fraction(video_stream.get("r_frame_rate")) or parse_fraction(
            video_stream.get("avg_frame_rate")
        )

    if timecode is None:
        for stream in streams:
            tags = stream.get("tags", {})
            codec_hint = (stream.get("codec_tag_string", "") + stream.get("codec_name", "")).lower()
            if tags.get("timecode") and (
                stream.get("codec_type") == "data" or "tmcd" in codec_hint
            ):
                timecode = tags.get("timecode")
                break

    resolved_timecode = timecode or fallback_timecode
    resolved_fps = fps or fallback_fps
    if resolved_timecode is None or resolved_fps is None:
        raise ValueError(
            "Could not resolve video timecode/fps automatically. "
            "Provide fallback_timecode and fallback_fps."
        )
    return resolved_timecode, float(resolved_fps), meta


def default_timecode_csv_path(video_file: Path) -> Path:
    return video_file.parent / f"video_timecode_1Hz{video_file.stem}.csv"


def default_metadata_json_path(video_file: Path) -> Path:
    return video_file.parent / f"video_metadata_{video_file.stem}.json"


def generate_video_timecode_csv(
    video_file: Path,
    output_csv: Path | None = None,
    metadata_json: Path | None = None,
    start_timecode: str | None = None,
    fps: float | None = None,
) -> tuple[Path, Path]:
    video_file = Path(video_file)
    output_csv = Path(output_csv) if output_csv is not None else default_timecode_csv_path(video_file)
    metadata_json = Path(metadata_json) if metadata_json is not None else default_metadata_json_path(video_file)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)

    start_timecode, fps, meta = resolve_timecode_and_fps(video_file, start_timecode, fps)
    hours, minutes, seconds, frames = map(int, start_timecode.replace(";", ":").split(":"))
    tc0_sec = hours * 3600 + minutes * 60 + seconds + frames / fps

    fmt = meta.get("format", {})
    duration = float(fmt["duration"])
    start_pts = float(fmt.get("start_time", 0) or 0.0)

    row_count = int(math.floor(duration))
    lines = [{"sec_from_start": sec} for sec in range(row_count + 1)]
    for row in lines:
        row["pts_time"] = start_pts + row["sec_from_start"]
        row["tc_seconds"] = tc0_sec + row["sec_from_start"]
        row["tc_str"] = sec_to_tc(row["tc_seconds"], fps)

    import pandas as pd

    pd.DataFrame(lines).to_csv(output_csv, index=False)
    metadata_payload = {
        "video_file": str(video_file),
        "timecode_csv": str(output_csv),
        "resolved_start_timecode": start_timecode,
        "resolved_fps": fps,
        "duration_seconds": duration,
        "start_pts_seconds": start_pts,
        "ffprobe": meta,
    }
    metadata_json.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    return output_csv, metadata_json


def ensure_video_timecode_csv(
    *,
    video_name: str | None = None,
    video_file: Path | str | None = None,
    videos_dir: Path | str | None = None,
    output_csv: Path | str | None = None,
    metadata_json: Path | str | None = None,
    fallback_timecode: str | None = None,
    fallback_fps: float | None = None,
    regenerate: bool = False,
) -> tuple[Path, Path, Path]:
    videos_dir = Path(videos_dir) if videos_dir is not None else Path.cwd() / "videos"
    resolved_video_file = Path(video_file) if video_file is not None else None
    if resolved_video_file is None:
        if not video_name:
            raise ValueError("Set either video_file or video_name.")
        resolved_video_file = find_video_file(video_name, videos_dir)

    output_csv = Path(output_csv) if output_csv is not None else default_timecode_csv_path(resolved_video_file)
    metadata_json = (
        Path(metadata_json) if metadata_json is not None else default_metadata_json_path(resolved_video_file)
    )

    if regenerate or not output_csv.exists() or not metadata_json.exists():
        generate_video_timecode_csv(
            resolved_video_file,
            output_csv=output_csv,
            metadata_json=metadata_json,
            start_timecode=fallback_timecode,
            fps=fallback_fps,
        )
    return resolved_video_file, output_csv, metadata_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract video LTC/timecode metadata and export 1 Hz CSV.")
    parser.add_argument("--video-name", help="Video stem used to find the source file inside --videos-dir.")
    parser.add_argument("--video-file", type=Path, help="Absolute or relative path to the source video.")
    parser.add_argument("--videos-dir", type=Path, default=Path.cwd() / "videos")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--fallback-timecode")
    parser.add_argument("--fallback-fps", type=float)
    parser.add_argument("--regenerate", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    video_file, output_csv, metadata_json = ensure_video_timecode_csv(
        video_name=args.video_name,
        video_file=args.video_file,
        videos_dir=args.videos_dir,
        output_csv=args.output_csv,
        metadata_json=args.metadata_json,
        fallback_timecode=args.fallback_timecode,
        fallback_fps=args.fallback_fps,
        regenerate=args.regenerate,
    )
    print(f"Video file: {video_file}")
    print(f"Timecode CSV: {output_csv}")
    print(f"Metadata JSON: {metadata_json}")


if __name__ == "__main__":
    main()
