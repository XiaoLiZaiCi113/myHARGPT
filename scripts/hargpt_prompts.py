from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .hargpt_data import PreparedData, load_cup_acc_csv, resample_acc


REFERENCE_TRAIN_RATIO = 0.70
MAX_REFERENCE_SAMPLES_PER_CLASS = 8


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


def select_evenly_spaced(items, max_count):
    if len(items) <= max_count:
        return items
    indices = np.linspace(0, len(items) - 1, max_count, dtype=int)
    return [items[int(index)] for index in indices]


def build_reference_samples(prepared: PreparedData, include_tilt=True):
    config = prepared.config
    if config.manual_label_output is None:
        raise ValueError("Manual label output path has not been initialized. Run prepare_video_aligned_data first.")
    if not config.manual_label_output.exists():
        raise FileNotFoundError(
            f"Manual label CSV not found: {config.manual_label_output}. "
            "Run build_weak_label_csv before generating reference-guided prompts."
        )

    manual_df = pd.read_csv(config.manual_label_output)
    manual_df["time"] = pd.to_datetime(manual_df["time"])
    manual_df = manual_df.sort_values("time").reset_index(drop=True)
    train_end = int(len(manual_df) * REFERENCE_TRAIN_RATIO)
    train_df = manual_df.iloc[:train_end].copy().reset_index(drop=True)
    if train_df.empty:
        return [], manual_df["time"].iloc[0]

    tilt_values = None
    if include_tilt:
        acc_full = resample_acc(load_cup_acc_csv(config.acc_file), config.target_fs)
        tilt_ref_samples = min(int(2 * config.target_fs), len(acc_full))
        full_tilt = tilt_angle_deg(
            acc_full["x"].to_numpy(dtype=float),
            acc_full["y"].to_numpy(dtype=float),
            acc_full["z"].to_numpy(dtype=float),
            ref_slice=slice(0, tilt_ref_samples),
        )
        video_start_in_full = int(acc_full["time"].searchsorted(prepared.acc_video.loc[0, "time"]))
        tilt_values = full_tilt[video_start_in_full : video_start_in_full + len(manual_df)]
        train_df["tilt"] = tilt_values[:train_end]

    chunk_size = max(1, int(round(config.window_seconds * config.target_fs)))
    reference_samples = []
    run_start = 0
    reference_id = 0
    for index in range(1, len(train_df) + 1):
        if index == len(train_df) or train_df.loc[index, "manual_label"] != train_df.loc[index - 1, "manual_label"]:
            run_label = str(train_df.loc[index - 1, "manual_label"])
            for chunk_start in range(run_start, index, chunk_size):
                chunk_end = min(chunk_start + chunk_size, index)
                chunk_df = train_df.iloc[chunk_start:chunk_end].copy()
                if chunk_df.empty:
                    continue
                raw_xyz_sequence = {
                    axis: [round(float(value), 6) for value in chunk_df[axis]]
                    for axis in ["x", "y", "z"]
                }
                segment_slices = {"full": slice(0, len(chunk_df))}
                segment_slices.update(three_way_slices(len(chunk_df)))
                statistic_axes = ["x", "y", "z"] + (["tilt"] if include_tilt and "tilt" in chunk_df else [])
                segment_statistics = {}
                for segment_name, segment_slice in segment_slices.items():
                    segment_statistics[segment_name] = {
                        axis: summarize_sequence(chunk_df[axis].to_numpy(dtype=float)[segment_slice])
                        for axis in statistic_axes
                    }
                sample = {
                    "reference_id": f"ref_{reference_id:04d}",
                    "label": run_label,
                    "start_time": str(chunk_df["time"].iloc[0]),
                    "end_time": str(chunk_df["time"].iloc[-1]),
                    "duration_s": round(float((chunk_df["time"].iloc[-1] - chunk_df["time"].iloc[0]).total_seconds()), 3),
                    "raw_xyz_sequence": raw_xyz_sequence,
                    "segment_statistics": segment_statistics,
                }
                if include_tilt and "tilt" in chunk_df:
                    sample["tilt_angle_deg_sequence"] = [round(float(value), 6) for value in chunk_df["tilt"]]
                reference_samples.append(sample)
                reference_id += 1
            run_start = index

    balanced_samples = []
    for label in ["Still", "Gesture", "Drinking", "Toasting", "Nodding"]:
        label_samples = [sample for sample in reference_samples if sample["label"] == label]
        balanced_samples.extend(select_evenly_spaced(label_samples, MAX_REFERENCE_SAMPLES_PER_CLASS))

    eval_start_time = manual_df["time"].iloc[train_end] if train_end < len(manual_df) else manual_df["time"].iloc[-1]
    return balanced_samples, eval_start_time


def build_reference_guided_payload(prepared: PreparedData, candidate_payloads, include_tilt=True):
    reference_samples, eval_start_time = build_reference_samples(prepared, include_tilt=include_tilt)
    window_index_df = prepared.window_index_df.copy()
    window_index_df["start_time"] = pd.to_datetime(window_index_df["start_time"])
    eval_window_ids = set(window_index_df.loc[window_index_df["start_time"] >= eval_start_time, "window_id"].astype(int))
    candidate_windows = [
        payload
        for payload in candidate_payloads
        if int(payload["window_id"]) in eval_window_ids
    ]
    return {
        "split": {
            "reference_train_ratio": REFERENCE_TRAIN_RATIO,
            "reference_source": "first 70% of sample-level manual labels",
            "candidate_source": "analysis windows starting in the final 30% of the video-aligned sample timeline",
            "eval_start_time": str(eval_start_time),
        },
        "reference_samples": reference_samples,
        "candidate_windows": candidate_windows,
    }


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


def build_reference_guided_hargpt_prompt(window_payloads, include_tilt=True):
    has_segment_statistics = False
    if isinstance(window_payloads, dict):
        candidate_windows = window_payloads.get("candidate_windows", [])
        has_segment_statistics = bool(candidate_windows) and "segment_statistics" in candidate_windows[0]
    else:
        has_segment_statistics = bool(window_payloads) and "segment_statistics" in window_payloads[0]

    input_fields = """- split: describes the chronological 70/30 reference/evaluation separation
- reference_samples: manually labeled reference segments from the first 70% of the sample timeline
  - reference_id: unique reference segment identifier
  - label: one of the candidate classes
  - raw_xyz_sequence: x-axis, y-axis, and z-axis acceleration sequences
  - segment_statistics: statistical summaries for full/start/mid/end"""
    if include_tilt:
        input_fields += "\n  - tilt_angle_deg_sequence: derived tilt angle sequence in degrees, when available"
    input_fields += "\n- candidate_windows: windows from the final 30% of the sample timeline to localize"
    input_fields += "\n  - window_id: unique window identifier"
    input_fields += "\n  - raw_xyz_sequence: x-axis, y-axis, and z-axis acceleration sequences"
    if include_tilt:
        input_fields += "\n  - tilt_angle_deg_sequence: derived tilt angle sequence in degrees"
    if has_segment_statistics:
        input_fields += "\n  - segment_statistics: statistical summaries for full/start/mid/end"

    return f"""You are an expert of smart-cup accelerometer based human activity recognition.

The device is a smart cup held by the user.
The sensor contains a three-axis accelerometer only.
The sampling rate is 50 Hz.
The data is represented in the smart-cup coordinate frame.

Your task is to localize activities inside each candidate window by comparing the candidate window with labeled reference samples from the same smart-cup setup.

Candidate classes, use these exact labels only:
- Still
- Gesture
- Drinking
- Toasting
- Nodding

Input JSON fields:
{input_fields}

Use the reference_samples as the primary evidence.
Compare each candidate window with the manually labeled reference samples using the available raw sequences and statistical summaries.
Base the prediction on similarity between the candidate window and reference-supported examples.
Do not invent new classes.
Do not rely on generic human activity assumptions when the reference samples provide stronger evidence.
If a candidate window contains multiple activity patterns, split it into temporal segments and assign each segment the closest matching reference-supported class.

For each segment, return:
- start_time
- end_time
- predicted_class
- reason

Use start_time and end_time in seconds relative to the beginning of the current candidate window.

Return exactly one JSON array with one object per candidate window:
[{{"window_id":0,"segments":[{{"start_time":0.00,"end_time":0.42,"predicted_class":"<class>","reason":"<brief explanation based on similarity to labeled reference samples>"}}]}}]

Return valid JSON only.
Do not use labels outside the five candidate classes."""


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
        raise ValueError(
            f"Start window {config.window_index_in_video_subset} is outside the available range 0..{max(len(window_index_df) - 1, 0)}."
        )

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

    batch_name = prepared.batch_name
    batch_dir = prepared.batch_dir
    batch_dir.mkdir(parents=True, exist_ok=True)

    feature_sets = {
        "xyz_RAG_tilt": {"payloads": window_payloads_xyz_rag_tilt(window_payloads), "include_tilt": True},
    }
    prompt_styles = {
        "Original_HARGPT": {"builder": build_original_hargpt_prompt, "prompt_file_suffix": "prompt_ori_loc"},
        "Rule_based": {"builder": build_rule_base_hargpt_prompt, "prompt_file_suffix": "prompt_rul_loc"},
        "Reference_guided": {
            "builder": build_reference_guided_hargpt_prompt,
            "prompt_file_suffix": "prompt_ref_loc",
            "payload_builder": build_reference_guided_payload,
        },
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
            if "payload_builder" in prompt_config:
                payloads = prompt_config["payload_builder"](prepared, payloads, include_tilt=include_tilt)
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
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
