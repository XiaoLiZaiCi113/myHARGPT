from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .hargpt_data import PreparedData, load_cup_acc_csv, resample_acc


def tilt_angle_deg(gx, gy, gz, ref_slice):
    gref = np.array([gx[ref_slice].mean(), gy[ref_slice].mean(), gz[ref_slice].mean()], dtype=float)
    gref_n = np.linalg.norm(gref) + 1e-8
    g = np.stack([gx, gy, gz], axis=1)
    g_n = np.linalg.norm(g, axis=1) + 1e-8
    cosv = (g @ gref) / (g_n * gref_n)
    cosv = np.clip(cosv, -1.0, 1.0)
    return np.degrees(np.arccos(cosv))


def summarize_sequence(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": None, "max": None, "min": None, "q1": None, "q3": None, "std": None, "median": None}
    return {
        "mean": round(float(np.mean(arr)), 6),
        "max": round(float(np.max(arr)), 6),
        "min": round(float(np.min(arr)), 6),
        "q1": round(float(np.percentile(arr, 25)), 6),
        "q3": round(float(np.percentile(arr, 75)), 6),
        "std": round(float(np.std(arr, ddof=0)), 6),
        "median": round(float(np.median(arr)), 6),
    }


def three_way_slices(length):
    edges = np.linspace(0, length, 4, dtype=int)
    return {"start": slice(edges[0], edges[1]), "mid": slice(edges[1], edges[2]), "end": slice(edges[2], edges[3])}


def window_payloads_xyz_only(window_payloads):
    return [{"window_id": window["window_id"], "raw_xyz_sequence": window["raw_xyz_sequence"]} for window in window_payloads]


def window_payloads_xyz_rag_tilt(window_payloads):
    rag_payloads = []
    for window in window_payloads:
        x_values = window["raw_xyz_sequence"]["x"]
        y_values = window["raw_xyz_sequence"]["y"]
        z_values = window["raw_xyz_sequence"]["z"]
        tilt_values = window["tilt_angle_deg_sequence"]
        segment_slices = {"full": slice(0, len(tilt_values))}
        segment_slices.update(three_way_slices(len(tilt_values)))
        segment_statistics = {}
        for segment_name, segment_slice in segment_slices.items():
            segment_statistics[segment_name] = {
                "x": summarize_sequence(x_values[segment_slice]),
                "y": summarize_sequence(y_values[segment_slice]),
                "z": summarize_sequence(z_values[segment_slice]),
                "tilt": summarize_sequence(tilt_values[segment_slice]),
            }
        rag_payloads.append(
            {
                "window_id": window["window_id"],
                "raw_xyz_sequence": window["raw_xyz_sequence"],
                "tilt_angle_deg_sequence": window["tilt_angle_deg_sequence"],
                "segment_statistics": segment_statistics,
            }
        )
    return rag_payloads


def build_original_hargpt_prompt(window_payloads, include_tilt=True):
    has_segment_statistics = bool(window_payloads) and "segment_statistics" in window_payloads[0]
    sensor_description = (
        "For each window, the three-axis accelerations, derived tilt angle, and RAG-style segment summaries for full/start/mid/end are given in the accompanying JSON input."
        if include_tilt and has_segment_statistics
        else "For each window, the three-axis accelerations and derived tilt angle are given in the accompanying JSON input."
        if include_tilt
        else "For each window, only the three-axis accelerations are given in the accompanying JSON input."
    )
    input_fields = "- window_id: unique window identifier\n- raw_xyz_sequence: x-axis, y-axis, and z-axis acceleration sequences"
    if include_tilt:
        input_fields += "\n- tilt_angle_deg_sequence: derived tilt angle sequence in degrees"
    if has_segment_statistics:
        input_fields += "\n- segment_statistics: statistical summaries for full/start/mid/end, each containing x/y/z/tilt mean, max, min, q1, q3, std, and median"

    return f"""You are an expert of IMU-based human activity analysis.

Question: The IMU data is collected from a smart cup held by the user with a sampling rate of 50 Hz. The IMU data is given in the smart-cup IMU coordinate frame. {sensor_description}

Input JSON fields:
{input_fields}

The person’s action belongs to one of the following categories:
- Still
- Gesture
- Drinking
- Toasting
- Nodding

Could you please tell me what action the person was doing based on the given
information and IMU readings?

Within each window, identify one or more temporal segments.
For each segment, return:
- start_time
- end_time
- predicted_class
- reason

Use start_time and end_time in seconds relative to the beginning of the current window.

Return exactly one JSON array with one object per window:
[{{"window_id":0,"segments":[{{"start_time":0.00,"end_time":0.42,"predicted_class":"<class>","reason":"<brief explanation>"}}]}}]

Use only the category labels listed above. Return in JSON file format."""


def build_rule_base_hargpt_prompt(window_payloads, include_tilt=True):
    has_segment_statistics = bool(window_payloads) and "segment_statistics" in window_payloads[0]
    input_fields = "- window_id: unique window identifier\n- raw_xyz_sequence: x-axis, y-axis, and z-axis acceleration sequences"
    if include_tilt:
        input_fields += "\n- tilt_angle_deg_sequence: derived tilt angle sequence in degrees"
    if has_segment_statistics:
        input_fields += "\n- segment_statistics: statistical summaries for full/start/mid/end, each containing x/y/z/tilt mean, max, min, q1, q3, std, and median"

    return f"""You are an expert of IMU-based human activity analysis.

I will give you raw motion sensor data from a smart cup.
The device contains an accelerometer only.
The device moves with the cup rather than being worn on the body.
The sampling rate is 50 Hz.

Your task is to localize the activities within each window from the raw sensor sequences.

Input JSON fields:
{input_fields}

Candidate classes (use these exact labels):
- Still
- Gesture
- Toasting
- Drinking
- Nodding

Please analyze the raw accelerometer data step by step.

Important distinction between Still and Nodding:
- Still does NOT mean simply low overall variance.
- Still should be used only when the signal shows no meaningful structured micro-movement and no nod-like localized event.
- Nodding can be subtle in a smart-cup accelerometer and may occur inside an otherwise still window.
- Do NOT require large amplitude or large tilt for Nodding.
- A Nodding event is typically a brief localized micro-event, often about 0.2 to 1.2 s, with a small dip-and-return, rise-and-return, or 1 to 3 short oscillatory pulses, followed by recovery toward the previous baseline.
- Judge Nodding by local structure relative to the nearby baseline, not only by absolute motion magnitude.

Decision order for each window:
1. First look for localized events, especially subtle Nodding segments, even if the whole window looks mostly quiet.
2. Only after checking for localized Nodding, decide whether the remaining parts are Still.
3. Prefer "Still + short Nodding segment" over labeling the entire window as Still when a plausible nod-like micro-pattern is present.
4. Use Gesture for broader non-drinking motion that is clearly structured but not better explained as Still, Nodding, Toasting, or Drinking.

Use these cues:
- overall motion intensity
- variability of acceleration
- relationships between x, y, and z axes
- tilt angle
- local departures from baseline
- brief oscillatory or dip-return structure
- recovery back toward baseline

Nodding positive cues:
- brief localized event within a mostly stable window
- small but structured deviation from baseline
- dip-and-return or short oscillatory pattern
- recovery toward the previous baseline
- subtle coordinated variation across axes or tilt

Nodding negative cues:
- completely flat signal with no localized deviation
- isolated noisy spike with no structure
- long broad movement better explained as Gesture or Drinking
- large monotonic tilt increase toward the mouth
- sustained high-angle hold
Do not default to a single full-window Still segment.
If the window is mostly quiet, explicitly check whether a short Nodding segment should be carved out from the quiet background.

Gesture means broader non-drinking hand/cup motion that is not better explained as Toasting, Drinking, or Nodding.
For drink consider only large angles (>40).
Drinking may contain a raise-to-mouth phase, an at-mouth or drinking-hold phase, and a return-from-mouth phase, but all of these should be labeled as Drinking.
Use those three drinking sub-phases only as reasoning cues for temporal structure, not as output labels.

Use temporal continuity across adjacent windows if it helps, but output the result separately for each window.

Within each window, identify one or more temporal segments.
For each segment, return:
- start_time
- end_time
- predicted_class
- reason

Use start_time and end_time in seconds relative to the beginning of the current window.

Return exactly one JSON array with one object per window:
[{{"window_id":0,"segments":[{{"start_time":0.00,"end_time":0.42,"predicted_class":"<class>","reason":"<brief explanation>"}}]}}]

Return in JSON file format.
Do not use code script to extract statistical values and still process the data directly.
"""


@dataclass
class PromptArtifacts:
    prepared: PreparedData
    window_index_df: pd.DataFrame
    acc_video: pd.DataFrame
    window_payloads: list[dict]
    batch_name: str
    batch_dir: Path
    feature_sets: dict[str, dict]
    prompt_styles: dict[str, dict]
    generated_files: list[Path]


def generate_prompt_artifacts(prepared: PreparedData) -> PromptArtifacts:
    config = prepared.config
    window_index_df = pd.read_csv(config.window_index_csv)
    window_index_df["start_time"] = pd.to_datetime(window_index_df["start_time"])
    window_index_df["end_time"] = pd.to_datetime(window_index_df["end_time"])
    acc_video = pd.read_csv(config.acc_video_csv)
    acc_video["time"] = pd.to_datetime(acc_video["time"])

    available_window_count = max(0, len(window_index_df) - config.window_index_in_video_subset)
    actual_prompt_window_count = min(config.prompt_window_count, available_window_count)
    if actual_prompt_window_count <= 0:
        raise ValueError(f"Start window {config.window_index_in_video_subset} is outside the available range 0..{max(len(window_index_df) - 1, 0)}.")

    selected_meta = window_index_df.iloc[
        config.window_index_in_video_subset : config.window_index_in_video_subset + actual_prompt_window_count
    ].copy()

    acc_full = resample_acc(load_cup_acc_csv(config.acc_file), config.target_fs)
    tilt_ref_samples = min(int(2 * config.target_fs), len(acc_full))
    if tilt_ref_samples == 0:
        raise ValueError("ACC file is empty; cannot compute tilt reference.")
    full_tilt = tilt_angle_deg(
        acc_full["x"].to_numpy(dtype=float),
        acc_full["y"].to_numpy(dtype=float),
        acc_full["z"].to_numpy(dtype=float),
        ref_slice=slice(0, tilt_ref_samples),
    )
    video_start_in_full = int(acc_full["time"].searchsorted(acc_video.loc[0, "time"]))

    window_payloads = []
    for row in selected_meta.itertuples(index=False):
        start_idx = int(row.start_idx_in_video_subset)
        end_idx = int(row.end_idx_in_video_subset) + 1
        window_df = acc_video.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        full_start_idx = video_start_in_full + start_idx
        full_end_idx = video_start_in_full + end_idx
        tilt = full_tilt[full_start_idx:full_end_idx]
        window_payloads.append(
            {
                "window_id": int(row.window_id),
                "raw_xyz_sequence": {
                    "x": [round(float(v), 6) for v in window_df["x"]],
                    "y": [round(float(v), 6) for v in window_df["y"]],
                    "z": [round(float(v), 6) for v in window_df["z"]],
                },
                "tilt_angle_deg_sequence": [round(float(v), 6) for v in tilt],
            }
        )

    actual_window_end = int(selected_meta["window_id"].iloc[-1])
    batch_name = f"hargpt_windows_{config.window_index_in_video_subset}_to_{actual_window_end}_{config.participant_id}_{config.video_name}"
    batch_dir = config.output_dir / batch_name
    batch_dir.mkdir(exist_ok=True)

    feature_sets = {
        "xyz_only": {"payloads": window_payloads_xyz_only(window_payloads), "include_tilt": False},
        "xyz_tilt": {"payloads": window_payloads, "include_tilt": True},
        "xyz_RAG_tilt": {"payloads": window_payloads_xyz_rag_tilt(window_payloads), "include_tilt": True},
    }
    prompt_styles = {
        "original HARGPT": {"builder": build_original_hargpt_prompt, "prompt_file_suffix": "prompt_window_classification"},
        "Rule_based_localization": {"builder": build_rule_base_hargpt_prompt, "prompt_file_suffix": "prompt_window_localization"},
    }

    generated_files = []
    for prompt_style, prompt_config in prompt_styles.items():
        style_dir = batch_dir / prompt_style
        for feature_set, feature_config in feature_sets.items():
            data_dir = style_dir / "window_data" / feature_set
            prompt_dir = style_dir / "prompts" / feature_set
            prediction_dir = style_dir / "predictions" / feature_set
            data_dir.mkdir(parents=True, exist_ok=True)
            prompt_dir.mkdir(parents=True, exist_ok=True)
            prediction_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = data_dir / f"{batch_name}_{feature_set}.json"
            prompt_path = prompt_dir / f"{batch_name}_{prompt_config['prompt_file_suffix']}_{feature_set}.txt"
            payloads = feature_config["payloads"]
            include_tilt = feature_config["include_tilt"]
            dataset_path.write_text(json.dumps(payloads, indent=2, ensure_ascii=False), encoding="utf-8")
            prompt_path.write_text(prompt_config["builder"](payloads, include_tilt=include_tilt), encoding="utf-8")
            generated_files.extend([dataset_path, prompt_path])

    return PromptArtifacts(prepared, window_index_df, acc_video, window_payloads, batch_name, batch_dir, feature_sets, prompt_styles, generated_files)


def prediction_dir_for(artifacts: PromptArtifacts, prompt_style: str, feature_set: str) -> Path:
    if prompt_style not in artifacts.prompt_styles:
        raise ValueError(f"Unknown prompt style: {prompt_style}. Use one of {list(artifacts.prompt_styles)}")
    if feature_set not in artifacts.feature_sets:
        raise ValueError(f"Unknown feature set: {feature_set}. Use one of {list(artifacts.feature_sets)}")
    path = artifacts.batch_dir / prompt_style / "predictions" / feature_set
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_prediction_jsons(artifacts: PromptArtifacts):
    for prompt_style in artifacts.prompt_styles:
        for feature_set in artifacts.feature_sets:
            prediction_dir = artifacts.batch_dir / prompt_style / "predictions" / feature_set
            if not prediction_dir.exists():
                continue
            for result_path in sorted(prediction_dir.glob("*.json")):
                yield prompt_style, feature_set, result_path


def summarize_prompt_artifacts(artifacts: PromptArtifacts) -> None:
    print(f"Prompt-style batch folder: {artifacts.batch_dir}")
    for generated_file in artifacts.generated_files:
        print(f"Generated: {generated_file}")
