from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from .hargpt_labels import LABEL_PRIORITY, LABEL_TO_STATE, STATE_COLORS, STATE_TO_NAME
from .hargpt_prompts import PromptArtifacts, iter_prediction_jsons


FINE_CLASS_TO_COARSE_LABEL = {
    "still": "Still",
    "gesture": "Gesture",
    "gesture stroke": "Gesture",
    "gesture phrase": "Gesture",
    "hand switch": "Gesture",
    "drinking": "Drinking",
    "drink": "Drinking",
    "raise to mouth": "Drinking",
    "at mouth or drinking hold": "Drinking",
    "return from mouth": "Drinking",
    "raise to lips": "Drinking",
    "return from lips": "Drinking",
    "toasting": "Toasting",
    "toast": "Toasting",
    "nodding": "Nodding",
    "head nod": "Nodding",
}


def normalize_label(value):
    if pd.isna(value):
        return pd.NA
    return str(value).strip().lower()


def map_prediction_to_coarse_label(value):
    normalized = normalize_label(value)
    if pd.isna(normalized):
        return pd.NA
    if normalized in FINE_CLASS_TO_COARSE_LABEL:
        return FINE_CLASS_TO_COARSE_LABEL[normalized]
    title_label = str(value).strip().title()
    if title_label in LABEL_TO_STATE:
        return title_label
    return pd.NA


def prediction_df_from_json(result_path: Path, window_index_df: pd.DataFrame) -> pd.DataFrame:
    predictions = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError("Prediction JSON must be a JSON array.")
    pred_df = pd.DataFrame(predictions)
    if pred_df.empty:
        raise ValueError("Prediction JSON is empty.")

    if "segments" in pred_df.columns and "predicted_class" not in pred_df.columns:
        pred_df = pred_df.explode("segments", ignore_index=True)
        seg_df = pd.json_normalize(pred_df["segments"]).rename(columns={"start_time": "segment_start_s", "end_time": "segment_end_s"})
        pred_df = pd.concat([pred_df.drop(columns=["segments"]).reset_index(drop=True), seg_df.reset_index(drop=True)], axis=1)
        required_cols = {"window_id", "segment_start_s", "segment_end_s", "predicted_class"}
        missing_cols = required_cols - set(pred_df.columns)
        if missing_cols:
            raise ValueError(f"Prediction JSON missing required fields: {sorted(missing_cols)}")
        pred_df = pred_df.merge(
            window_index_df[["window_id", "start_time", "end_time"]].rename(columns={"start_time": "window_start_time", "end_time": "window_end_time"}),
            on="window_id",
            how="left",
        )
        pred_df["start_time"] = pred_df["window_start_time"] + pd.to_timedelta(pred_df["segment_start_s"], unit="s")
        pred_df["end_time"] = pred_df["window_start_time"] + pd.to_timedelta(pred_df["segment_end_s"], unit="s")
        pred_df["start_time"] = pred_df[["start_time", "window_start_time"]].max(axis=1)
        pred_df["end_time"] = pred_df[["end_time", "window_end_time"]].min(axis=1)
    else:
        required_cols = {"window_id", "predicted_class"}
        missing_cols = required_cols - set(pred_df.columns)
        if missing_cols:
            raise ValueError(f"Prediction JSON missing required fields: {sorted(missing_cols)}")
        pred_df = pred_df.merge(window_index_df[["window_id", "start_time", "end_time"]], on="window_id", how="left")

    pred_df["coarse_label"] = pred_df["predicted_class"].map(map_prediction_to_coarse_label)
    pred_df["predicted_state"] = pred_df["coarse_label"].map(LABEL_TO_STATE)
    pred_df = pred_df.sort_values(["window_id", "start_time", "end_time"]).reset_index(drop=True)
    invalid_rows = pred_df[pred_df[["coarse_label", "predicted_state", "start_time", "end_time"]].isna().any(axis=1)].copy()
    if not invalid_rows.empty:
        pred_df = pred_df.drop(index=invalid_rows.index).reset_index(drop=True)
    pred_df = pred_df[pred_df["end_time"] >= pred_df["start_time"]].reset_index(drop=True)
    if pred_df.empty:
        raise ValueError("No valid prediction intervals available after validation.")
    pred_df["predicted_state"] = pred_df["predicted_state"].astype(int)
    return pred_df


def plot_prediction_result(artifacts: PromptArtifacts, manual_df: pd.DataFrame, prompt_style: str, feature_set: str, result_path: Path) -> pd.DataFrame:
    pred_df = prediction_df_from_json(result_path, artifacts.window_index_df)
    output_dir = result_path.parent / result_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    compare_png = output_dir / "comparison_vs_manual_labels.png"
    confusion_csv = output_dir / "confusion_counts.csv"

    plot_start = pred_df["start_time"].min()
    plot_end = pred_df["end_time"].max()
    plot_origin = plot_start
    acc_plot = artifacts.acc_video[(artifacts.acc_video["time"] >= plot_start) & (artifacts.acc_video["time"] <= plot_end)].copy().reset_index(drop=True)
    manual_plot = manual_df[(manual_df["time"] >= plot_start) & (manual_df["time"] <= plot_end)].copy().reset_index(drop=True)
    if manual_plot.empty:
        raise ValueError("No manual labels overlap with the selected prediction range.")

    manual_plot = manual_plot.sort_values("time").reset_index(drop=True).rename(columns={"state": "manual_state"})
    manual_plot["manual_state"] = manual_plot["manual_state"].astype(int)
    acc_pred = manual_plot[["time", "manual_state", "manual_label"]].copy()
    acc_pred["predicted_state"] = LABEL_TO_STATE["Still"]
    acc_pred["predicted_label"] = "Still"
    priority_buffer = pd.Series(LABEL_PRIORITY["Still"], index=acc_pred.index, dtype=int)

    for row in pred_df.itertuples(index=False):
        coarse_label = row.coarse_label
        priority = LABEL_PRIORITY[coarse_label]
        mask = (acc_pred["time"] >= row.start_time) & (acc_pred["time"] <= row.end_time)
        if not mask.any():
            continue
        update_mask = mask & (priority >= priority_buffer)
        if not update_mask.any():
            continue
        priority_buffer.loc[update_mask] = priority
        acc_pred.loc[update_mask, "predicted_state"] = LABEL_TO_STATE[coarse_label]
        acc_pred.loc[update_mask, "predicted_label"] = coarse_label

    acc_pred["time_rel_s"] = (acc_pred["time"] - plot_origin).dt.total_seconds()
    acc_plot["time_rel_s"] = (acc_plot["time"] - plot_origin).dt.total_seconds()

    manual_runs = []
    start_idx = 0
    for i in range(1, len(acc_pred) + 1):
        if i == len(acc_pred) or acc_pred.loc[i, "manual_label"] != acc_pred.loc[i - 1, "manual_label"]:
            manual_runs.append((acc_pred.loc[start_idx, "time_rel_s"], acc_pred.loc[i - 1, "time_rel_s"], acc_pred.loc[i - 1, "manual_label"]))
            start_idx = i
    predicted_runs = []
    start_idx = 0
    for i in range(1, len(acc_pred) + 1):
        if i == len(acc_pred) or acc_pred.loc[i, "predicted_label"] != acc_pred.loc[i - 1, "predicted_label"]:
            predicted_runs.append((acc_pred.loc[start_idx, "time_rel_s"], acc_pred.loc[i - 1, "time_rel_s"], acc_pred.loc[i - 1, "predicted_label"]))
            start_idx = i

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax0, ax1 = axes
    for ax in (ax0, ax1):
        ax.tick_params(axis="both", colors="black", labelcolor="black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.title.set_color("black")
        for spine in ax.spines.values():
            spine.set_color("black")
    ax0.plot(acc_plot["time_rel_s"], acc_plot["x"], label="x", color="blue", linewidth=0.7)
    ax0.plot(acc_plot["time_rel_s"], acc_plot["y"], label="y", color="red", linewidth=0.7)
    ax0.plot(acc_plot["time_rel_s"], acc_plot["z"], label="z", color="black", linewidth=0.7)
    for start_time, end_time, predicted_label in predicted_runs:
        predicted_state = LABEL_TO_STATE.get(str(predicted_label), 0)
        ax0.axvspan(start_time, end_time, color=STATE_COLORS[int(predicted_state)], alpha=0.18)
    ax0.set_title(f"ACC with predictions: {prompt_style} / {feature_set} / {result_path.stem}")
    ax0.set_ylabel("Acceleration")
    ax0.legend(loc="upper right")
    for start_time, end_time, predicted_label in predicted_runs:
        predicted_state = LABEL_TO_STATE.get(str(predicted_label), 0)
        ax1.axvspan(start_time, end_time, ymin=0.52, ymax=0.98, color=STATE_COLORS[int(predicted_state)], alpha=0.45)
    for start_time, end_time, manual_label in manual_runs:
        manual_state = LABEL_TO_STATE.get(str(manual_label), 0)
        ax1.axvspan(start_time, end_time, ymin=0.02, ymax=0.48, color=STATE_COLORS[int(manual_state)], alpha=0.45)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0.25, 0.75])
    ax1.set_yticklabels(["Manual label", "Prediction"])
    ax1.set_title("Manual labels vs predictions")
    ax1.set_xlabel("Plot relative time (s)")
    legend_handles = [Patch(facecolor=STATE_COLORS[s], edgecolor="gray", label=STATE_TO_NAME[s]) for s in [0, 1, 2, 3, 4]]
    ax1.legend(handles=legend_handles, loc="upper right")
    fig.tight_layout()
    fig.savefig(compare_png, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="white", transparent=False)
    plt.close(fig)

    state_order = [0, 1, 2, 3, 4]
    state_names = [STATE_TO_NAME[state] for state in state_order]
    manual_cat = pd.Categorical(acc_pred["manual_state"], categories=state_order, ordered=True)
    pred_cat = pd.Categorical(acc_pred["predicted_state"], categories=state_order, ordered=True)
    confusion_counts = pd.crosstab(manual_cat, pred_cat, dropna=False)
    confusion_counts.index = [f"manual:{name}" for name in state_names]
    confusion_counts.columns = [f"prediction:{name}" for name in state_names]
    true_positive_count = int(sum(confusion_counts.iloc[i, i] for i in range(len(state_order))))
    total_population = int(confusion_counts.to_numpy().sum())
    total_accuracy = true_positive_count / total_population if total_population else 0.0
    confusion_counts.to_csv(confusion_csv)
    with confusion_csv.open("a", encoding="utf-8", newline="") as file:
        file.write("\nmetric,value\n")
        file.write(f"true_positive_diagonal_sum,{true_positive_count}\n")
        file.write(f"total_population,{total_population}\n")
        file.write(f"total_accuracy,{total_accuracy:.6f}\n")

    print(f"Saved comparison plot to: {compare_png}")
    print(f"Saved confusion counts to: {confusion_csv}")
    print(f"Total accuracy = {true_positive_count} / {total_population} = {total_accuracy:.6f}")
    print(f"Confusion matrix counts for {prompt_style} / {feature_set} / {result_path.name}:")
    display(confusion_counts)
    return acc_pred[["time", "manual_label", "manual_state", "predicted_label", "predicted_state"]].head()


def evaluate_all_predictions(artifacts: PromptArtifacts) -> dict[tuple[str, str, str], pd.DataFrame]:
    if artifacts.prepared.config.manual_label_output is None:
        raise ValueError("Manual label output path has not been initialized. Run prepare_video_aligned_data first.")
    manual_df = pd.read_csv(artifacts.prepared.config.manual_label_output)
    manual_df["time"] = pd.to_datetime(manual_df["time"])
    prediction_results = list(iter_prediction_jsons(artifacts))
    if not prediction_results:
        raise FileNotFoundError(f"No prediction JSON files found under {artifacts.batch_dir}/*/predictions/*/")
    plot_heads = {}
    for prompt_style, feature_set, result_path in prediction_results:
        print(f"Using prediction JSON: {result_path}")
        try:
            plot_heads[(prompt_style, feature_set, result_path.name)] = plot_prediction_result(artifacts, manual_df, prompt_style, feature_set, result_path)
        except (json.JSONDecodeError, ValueError, KeyError, TypeError, OSError) as exc:
            print(f"Skipping {result_path}: {exc}")
    if not plot_heads:
        raise ValueError("No valid prediction JSON files were found for plotting.")
    return plot_heads
