from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .covfee_preprocessing import preprocess_covfee_annotations


CHANNEL_COLUMNS = ["x", "y", "z"]
TARGET_COLUMN = "manual_label"
BACKGROUND_LABEL = "Still"
CANONICAL_LABEL_ORDER = ["Still", "Gesture", "Nodding", "Drinking", "Toasting"]
SPLIT_CONFIGS = {
    "split_A_eval_01_v1": {"eval_annotations": ["01_v1", "20_v2", "27_v2"]},
    "split_B_eval_24_v1": {"eval_annotations": ["24_v1", "18_v2", "27_v2"]},
}


def load_covfee_manual_data(
    project_root: Path,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    project_root = Path(project_root).resolve()
    output_dir = output_dir if output_dir is not None else project_root / "outputs" / "covfee"
    combined, summary = preprocess_covfee_annotations(project_root, output_dir=output_dir)
    data_path = output_dir / "covfee_manual_labels_combined.csv"
    if combined.empty:
        raise ValueError("COVFEE preprocessing did not produce any labeled ACC rows.")
    df = pd.read_csv(data_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["annotation_id", "time"]).reset_index(drop=True)
    return df, summary, data_path


def estimate_sampling_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for annotation_id, group_df in df.groupby("annotation_id", sort=False):
        diffs = group_df["time"].diff().dt.total_seconds().dropna()
        diffs = diffs[diffs > 0]
        if diffs.empty:
            continue
        rows.append(
            {
                "annotation_id": annotation_id,
                "estimated_fs": 1.0 / diffs.median(),
                "rows": len(group_df),
            }
        )
    return pd.DataFrame(rows)


def maybe_downsample(frame: pd.DataFrame, original_fs: float, target_fs: float | None) -> pd.DataFrame:
    if target_fs is None or target_fs >= original_fs:
        return frame.copy()
    step = max(1, int(round(original_fs / target_fs)))
    return frame.iloc[::step].reset_index(drop=True)


def build_windows_for_group(
    frame: pd.DataFrame,
    sampling_rate: float,
    window_seconds: float,
    stride_seconds: float,
    channel_columns: list[str],
    target_column: str = TARGET_COLUMN,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, int, int]:
    window_size = int(round(window_seconds * sampling_rate))
    stride_size = int(round(stride_seconds * sampling_rate))
    x_windows = []
    y_windows = []
    meta_rows = []

    for start in range(0, len(frame) - window_size + 1, stride_size):
        end = start + window_size
        window = frame.iloc[start:end]
        label_distribution = window[target_column].value_counts(normalize=True)
        majority_label = window[target_column].mode().iloc[0]
        x_windows.append(window[channel_columns].to_numpy(dtype=np.float32).T)
        y_windows.append(majority_label)
        meta_rows.append(
            {
                "start_idx": start,
                "end_idx": end,
                "start_time": window["time"].iloc[0],
                "end_time": window["time"].iloc[-1],
                "label": majority_label,
                "purity": float(label_distribution.iloc[0]),
                "n_unique_labels": int(label_distribution.shape[0]),
                "annotation_id": window["annotation_id"].iloc[0],
                "participant_no": window["participant_no"].iloc[0],
                "participant_id": window["participant_id"].iloc[0],
                "video_name": window["video_name"].iloc[0],
                "alignment_mode": window["alignment_mode"].iloc[0],
            }
        )

    if not x_windows:
        return np.empty((0, len(channel_columns), window_size), dtype=np.float32), np.asarray([]), pd.DataFrame(), window_size, stride_size
    return np.stack(x_windows), np.asarray(y_windows), pd.DataFrame(meta_rows), window_size, stride_size


def build_grouped_windows(
    df: pd.DataFrame,
    *,
    channel_columns: list[str] = CHANNEL_COLUMNS,
    target_column: str = TARGET_COLUMN,
    target_fs: float | None = None,
    window_seconds: float = 1.0,
    stride_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, int, int]:
    x_parts = []
    y_parts = []
    meta_parts = []
    sampling_rows = []
    window_size = None
    stride_size = None

    for annotation_id, group_df in df.groupby("annotation_id", sort=False):
        diffs = group_df["time"].diff().dt.total_seconds().dropna()
        diffs = diffs[diffs > 0]
        if diffs.empty:
            continue
        original_fs = 1.0 / diffs.median()
        model_group_df = maybe_downsample(group_df, original_fs=original_fs, target_fs=target_fs)
        effective_fs = original_fs if target_fs is None else min(target_fs, original_fs)
        x_group, y_group, meta_group, group_window_size, group_stride_size = build_windows_for_group(
            model_group_df,
            sampling_rate=effective_fs,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            channel_columns=channel_columns,
            target_column=target_column,
        )
        if len(y_group) == 0:
            continue
        meta_group["estimated_fs"] = effective_fs
        x_parts.append(x_group)
        y_parts.append(y_group)
        meta_parts.append(meta_group)
        sampling_rows.append({"annotation_id": annotation_id, "estimated_fs": effective_fs, "rows": len(model_group_df)})
        window_size = group_window_size if window_size is None else window_size
        stride_size = group_stride_size if stride_size is None else stride_size

    if not x_parts:
        raise ValueError("No windows could be built from the COVFEE data.")
    return (
        np.concatenate(x_parts, axis=0),
        np.concatenate(y_parts),
        pd.concat(meta_parts, ignore_index=True),
        pd.DataFrame(sampling_rows),
        int(window_size),
        int(stride_size),
    )


def filter_windows(
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    *,
    min_label_purity: float = 0.0,
    background_label: str = BACKGROUND_LABEL,
    drop_still_windows: bool = False,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, int]]:
    purity_mask = (meta["purity"] >= min_label_purity).to_numpy()
    background_mask = y == background_label
    selection_mask = purity_mask.copy()
    if drop_still_windows:
        selection_mask &= ~background_mask
    diagnostics = {
        "n_removed_by_purity": int((~purity_mask).sum()),
        "n_removed_background": int(np.sum(purity_mask & background_mask)) if drop_still_windows else 0,
    }
    x_selected = x[selection_mask]
    y_selected = y[selection_mask]
    meta_selected = meta.loc[selection_mask].reset_index(drop=True)
    if len(y_selected) == 0:
        raise ValueError("No windows left after purity/background filtering.")
    return x_selected, y_selected, meta_selected, diagnostics


def build_annotation_splits(
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    split_configs: dict[str, dict[str, list[str]]] = SPLIT_CONFIGS,
) -> dict[str, dict[str, object]]:
    available_annotations = set(meta["annotation_id"].unique())
    splits = {}
    for split_name, split_config in split_configs.items():
        eval_annotations = set(split_config["eval_annotations"])
        missing_annotations = sorted(eval_annotations - available_annotations)
        if missing_annotations:
            raise ValueError(f"{split_name} references missing annotations: {missing_annotations}")
        eval_mask = meta["annotation_id"].isin(eval_annotations).to_numpy()
        train_mask = ~eval_mask
        splits[split_name] = {
            "x_train": x[train_mask],
            "y_train": y[train_mask],
            "train_meta": meta.loc[train_mask].reset_index(drop=True),
            "x_eval": x[eval_mask],
            "y_eval": y[eval_mask],
            "eval_meta": meta.loc[eval_mask].reset_index(drop=True),
        }
        if len(splits[split_name]["y_train"]) == 0 or len(splits[split_name]["y_eval"]) == 0:
            raise ValueError(f"{split_name} produced an empty train or eval split.")
    return splits


def summarize_splits(
    splits: dict[str, dict[str, object]],
    label_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_summary_rows = []
    label_count_rows = []
    annotation_rows = []
    for split_name, split_data in splits.items():
        for subset_name, meta_key, y_key in [
            ("train", "train_meta", "y_train"),
            ("eval", "eval_meta", "y_eval"),
        ]:
            subset_meta = split_data[meta_key]
            subset_y = pd.Series(split_data[y_key])
            split_summary_rows.append(
                {
                    "split_config": split_name,
                    "subset": subset_name,
                    "n_windows": len(subset_meta),
                    "annotations": subset_meta["annotation_id"].nunique(),
                    "start": subset_meta["start_time"].min(),
                    "end": subset_meta["end_time"].max(),
                }
            )
            for label, count in subset_y.value_counts().reindex(label_order, fill_value=0).items():
                label_count_rows.append(
                    {
                        "split_config": split_name,
                        "subset": subset_name,
                        "label": label,
                        "n_windows": int(count),
                    }
                )
            for annotation_id, count in subset_meta["annotation_id"].value_counts().sort_index().items():
                annotation_rows.append(
                    {
                        "split_config": split_name,
                        "subset": subset_name,
                        "annotation_id": annotation_id,
                        "n_windows": int(count),
                    }
                )
    label_counts = pd.DataFrame(label_count_rows).pivot_table(
        index=["split_config", "label"],
        columns="subset",
        values="n_windows",
        fill_value=0,
    ).astype(int)
    annotation_counts = pd.DataFrame(annotation_rows).pivot_table(
        index=["split_config", "annotation_id"],
        columns="subset",
        values="n_windows",
        fill_value=0,
    ).astype(int)
    return pd.DataFrame(split_summary_rows), label_counts, annotation_counts


def label_order_from_values(values: np.ndarray | pd.Series) -> list[str]:
    present = set(np.asarray(values))
    return [label for label in CANONICAL_LABEL_ORDER if label in present]


def active_label_order(y_true: np.ndarray, y_pred: np.ndarray) -> list[str]:
    present = set(np.asarray(y_true)) | set(np.asarray(y_pred))
    return [label for label in CANONICAL_LABEL_ORDER if label != BACKGROUND_LABEL and label in present]


def evaluate_split(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_array,
        y_pred_array,
        average="macro",
        zero_division=0,
    )
    active_mask = y_true_array != BACKGROUND_LABEL
    active_accuracy = np.nan
    active_macro_precision = np.nan
    active_macro_recall = np.nan
    active_macro_f1 = np.nan
    active_n_samples = int(active_mask.sum())
    if active_n_samples:
        active_accuracy = accuracy_score(y_true_array[active_mask], y_pred_array[active_mask])
        labels_without_still = active_label_order(y_true_array[active_mask], y_pred_array[active_mask])
        if labels_without_still:
            active_macro_precision, active_macro_recall, active_macro_f1, _ = precision_recall_fscore_support(
                y_true_array[active_mask],
                y_pred_array[active_mask],
                labels=labels_without_still,
                average="macro",
                zero_division=0,
            )
    return {
        "split": name,
        "accuracy": accuracy_score(y_true_array, y_pred_array),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "active_accuracy": active_accuracy,
        "active_macro_precision_no_still": active_macro_precision,
        "active_macro_recall_no_still": active_macro_recall,
        "active_macro_f1_no_still": active_macro_f1,
        "n_samples": len(y_true_array),
        "active_n_samples": active_n_samples,
    }


def evaluate_timeline_diagnostics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    background_mask = y_true_array == BACKGROUND_LABEL
    active_mask = ~background_mask
    background_recall = np.nan
    false_activity_rate = np.nan
    if background_mask.any():
        background_recall = float(np.mean(y_pred_array[background_mask] == BACKGROUND_LABEL))
        false_activity_rate = 1.0 - background_recall
    active_accuracy = np.nan
    active_macro_f1 = np.nan
    if active_mask.any():
        active_accuracy = accuracy_score(y_true_array[active_mask], y_pred_array[active_mask])
        active_labels = active_label_order(y_true_array[active_mask], y_pred_array[active_mask])
        if active_labels:
            active_macro_f1 = precision_recall_fscore_support(
                y_true_array[active_mask],
                y_pred_array[active_mask],
                labels=active_labels,
                average="macro",
                zero_division=0,
            )[2]
    return {
        "split": name,
        "background_windows": int(background_mask.sum()),
        "active_windows": int(active_mask.sum()),
        "background_recall": background_recall,
        "false_activity_rate_on_background": false_activity_rate,
        "active_accuracy_on_active_windows": active_accuracy,
        "active_macro_f1_no_still_on_active_windows": active_macro_f1,
    }


def standardize_from_train(x_train: np.ndarray, *splits: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return tuple(((split - mean) / std).astype(np.float32) for split in splits)
