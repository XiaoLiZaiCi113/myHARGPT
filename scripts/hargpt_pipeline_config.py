from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PARTICIPANT_TO_ACCID = {
    1: 81, 2: 64, 3: 72, 4: 78, 5: 74, 6: 85, 7: 73, 8: 87,
    9: 92, 10: 71, 11: 84, 12: 95, 13: 86, 14: 76, 15: 67, 16: 68,
    17: 77, 18: 89, 19: 70, 20: 93, 21: 79, 22: 90, 23: 99, 24: 65,
    25: 75, 26: 80, 27: 66, 28: 82, 29: 83, 30: 94, 31: 91, 32: 88,
}


@dataclass
class PipelineConfig:
    project_root: Path
    participant_no: int = 24
    video_name: str = "GH040226"
    video_file: Path | None = None
    video_start_timecode: str | None = None
    video_fps: float | None = None
    force_regenerate_video_metadata: bool = False
    target_fs: float = 50.0
    window_seconds: float = 2.0
    stride_seconds: float = 2.0
    window_index_in_video_subset: int = 0
    prompt_window_count: int = 400

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.output_dir = self.project_root / "outputs"
        self.video_dir = self.project_root / "videos"
        self.output_dir.mkdir(exist_ok=True)
        self.video_dir.mkdir(exist_ok=True)

        self.participant_id = str(PARTICIPANT_TO_ACCID[self.participant_no])
        self.window_size = int(self.target_fs * self.window_seconds)
        self.stride_size = int(self.target_fs * self.stride_seconds)

        self.acc_file = self.project_root / "acc_files_cup" / self.participant_id / "ACC_0.csv"
        self.window_index_csv = self.output_dir / f"hargpt_window_index_{self.participant_id}_{self.video_name}.csv"
        self.acc_video_csv = self.output_dir / f"hargpt_acc_video_subset_{self.participant_id}_{self.video_name}.csv"
        self.weak_label_input = self.project_root / "weak_label" / f"{self.participant_no}.json"
        self.weak_label_output = self.project_root / "weak_label" / f"weak_labels_{self.participant_id}_{self.video_name}.csv"
        self.weak_label_overlay_png = self.output_dir / f"weak_labels_overlay_{self.participant_id}_{self.video_name}.png"
