"""
Analyze pairwise diversity metrics for the heterogeneous ensemble model pool.

Loads binarized predictions (threshold=0.5) from each base model and computes
four standard pairwise diversity measures plus per-model performance metrics.

Diversity measures (all pairwise, computed from 2x2 contingency tables):
  - Q-statistic (Yule's Q): agreement measure in [-1, 1]; Q=1 means identical
  - Disagreement: fraction of pixels where two models disagree
  - Correlation (phi coefficient): Pearson correlation of binary predictions
  - Double-fault: fraction of pixels where both models are wrong

Usage:
    pixi run python scripts/analyze_ensemble_diversity.py \
        --predictions_dir predictions \
        --split test \
        --output_dir diversity_analysis
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def load_predictions_and_gt(predictions_dir: Path, split: str, threshold: float = 0.5):
    """Load predictions and ground truth for all models.

    Returns:
        models_binary: dict of model_name -> binarized predictions (int8)
        models_soft: dict of model_name -> soft probability predictions (float32)
        gt: binarized ground truth (int8)
    """
    model_dirs = sorted(predictions_dir.iterdir())
    model_dirs = [d for d in model_dirs if d.is_dir()]

    models_binary = {}
    models_soft = {}
    gt = None

    for model_dir in model_dirs:
        pred_file = model_dir / f"{split}_predictions.pt"
        gt_file = model_dir / f"{split}_ground_truths.pt"

        if not pred_file.exists():
            continue

        preds = torch.load(pred_file, map_location="cpu", weights_only=True)
        model_name = model_dir.name

        models_soft[model_name] = preds.flatten().numpy().astype(np.float32)
        models_binary[model_name] = (preds > threshold).flatten().numpy().astype(np.int8)

        if gt is None:
            gt_raw = torch.load(gt_file, map_location="cpu", weights_only=True)
            gt = gt_raw.flatten().numpy().astype(np.int8)

    return models_binary, models_soft, gt


def compute_contingency(y1: np.ndarray, y2: np.ndarray, gt: np.ndarray):
    """
    Compute the 2x2 contingency table for a pair of classifiers.

    Returns (N11, N10, N01, N00) where:
        N11 = both correct
        N10 = model1 correct, model2 wrong
        N01 = model1 wrong, model2 correct
        N00 = both wrong (double fault)
    """
    correct1 = (y1 == gt)
    correct2 = (y2 == gt)

    n11 = np.sum(correct1 & correct2)
    n10 = np.sum(correct1 & ~correct2)
    n01 = np.sum(~correct1 & correct2)
    n00 = np.sum(~correct1 & ~correct2)

    return n11, n10, n01, n00


def q_statistic(n11, n10, n01, n00):
    """Yule's Q-statistic: (N11*N00 - N01*N10) / (N11*N00 + N01*N10)."""
    num = n11 * n00 - n01 * n10
    den = n11 * n00 + n01 * n10
    if den == 0:
        return 1.0
    return num / den


def disagreement_measure(n11, n10, n01, n00):
    """Fraction of samples where the two classifiers disagree."""
    total = n11 + n10 + n01 + n00
    return (n10 + n01) / total


def correlation_coefficient(n11, n10, n01, n00):
    """Phi coefficient (Pearson correlation of binary predictions)."""
    # Use float64 to avoid integer overflow with large pixel counts
    n11, n10, n01, n00 = float(n11), float(n10), float(n01), float(n00)
    num = n11 * n00 - n01 * n10
    den = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if den == 0:
        return 0.0
    return num / den


def double_fault_measure(n11, n10, n01, n00):
    """Fraction of samples where both classifiers are wrong."""
    total = n11 + n10 + n01 + n00
    return n00 / total


def compute_model_performance(models: dict, gt: np.ndarray):
    """Compute per-model performance metrics from binary predictions."""
    rows = []
    for name, preds in models.items():
        tp = np.sum((preds == 1) & (gt == 1))
        tn = np.sum((preds == 0) & (gt == 0))
        fp = np.sum((preds == 1) & (gt == 0))
        fn = np.sum((preds == 0) & (gt == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Strip numeric prefix for clean model name
        clean_name = name.split("_", 1)[1] if name[0].isdigit() else name
        rows.append({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "TP": float(tp),
            "TN": float(tn),
            "FP": float(fp),
            "FN": float(fn),
            "model": clean_name,
        })

    df = pd.DataFrame(rows).sort_values("f1").reset_index(drop=True)
    return df


def compute_pairwise_matrices(models_binary: dict, models_soft: dict, gt: np.ndarray):
    """Compute all pairwise diversity matrices.

    Q-statistic, disagreement, and double-fault use binarized predictions
    (standard ensemble diversity measures from Kuncheva 2003).
    Correlation uses soft (probability) predictions via Pearson correlation,
    capturing how similarly models rank pixels rather than just binary agreement.
    """
    names = list(models_binary.keys())
    n = len(names)

    q_mat = np.zeros((n, n))
    dis_mat = np.zeros((n, n))
    corr_mat = np.zeros((n, n))
    df_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                q_mat[i, j] = 1.0
                corr_mat[i, j] = 1.0
                continue
            if j < i:
                q_mat[i, j] = q_mat[j, i]
                dis_mat[i, j] = dis_mat[j, i]
                corr_mat[i, j] = corr_mat[j, i]
                df_mat[i, j] = df_mat[j, i]
                continue

            # Binary diversity measures
            n11, n10, n01, n00 = compute_contingency(
                models_binary[names[i]], models_binary[names[j]], gt
            )
            q_mat[i, j] = q_statistic(n11, n10, n01, n00)
            dis_mat[i, j] = disagreement_measure(n11, n10, n01, n00)
            df_mat[i, j] = double_fault_measure(n11, n10, n01, n00)

            # Soft prediction correlation (Pearson)
            corr_mat[i, j] = np.corrcoef(
                models_soft[names[i]], models_soft[names[j]]
            )[0, 1]

    matrices = {
        "q_statistic": q_mat,
        "disagreement": dis_mat,
        "correlation": corr_mat,
        "double_fault": df_mat,
    }
    return names, matrices


def compute_diversity_summary(names: list, matrices: dict):
    """Compute per-model summary statistics from pairwise matrices."""
    n = len(names)
    rows = []

    for i, name in enumerate(names):
        clean_name = name.split("_", 1)[1] if name[0].isdigit() else name
        # Exclude self-comparisons (diagonal)
        mask = [j for j in range(n) if j != i]

        q_vals = [matrices["q_statistic"][i, j] for j in mask]
        dis_vals = [matrices["disagreement"][i, j] for j in mask]
        corr_vals = [matrices["correlation"][i, j] for j in mask]
        df_vals = [matrices["double_fault"][i, j] for j in mask]

        rows.append({
            "model": clean_name,
            "avg_q_statistic": np.mean(q_vals),
            "avg_disagreement": np.mean(dis_vals),
            "avg_correlation": np.mean(corr_vals),
            "avg_double_fault": np.mean(df_vals),
            "min_q_statistic": np.min(q_vals),
            "max_disagreement": np.max(dis_vals),
        })

    return pd.DataFrame(rows).sort_values("avg_correlation").reset_index(drop=True)


def save_matrix_csv(names: list, matrix: np.ndarray, output_path: Path):
    """Save a pairwise matrix as CSV with model names as index/columns."""
    df = pd.DataFrame(matrix, index=names, columns=names)
    df.to_csv(output_path)


def plot_heatmap(names: list, matrix: np.ndarray, title: str, output_path: Path,
                 vmin=None, vmax=None, cmap="RdYlBu_r", fmt=".4f"):
    """Plot a pairwise matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    clean_names = [n.split("_", 1)[1] if n[0].isdigit() else n for n in names]

    sns.heatmap(
        matrix,
        xticklabels=clean_names,
        yticklabels=clean_names,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        square=True,
    )
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_model_performance(perf_df: pd.DataFrame, output_path: Path):
    """Scatter plot of F1 vs model (bar chart sorted by F1)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(perf_df["model"], perf_df["f1"])
    ax.set_xlabel("F1 Score")
    ax.set_title("Model Performance (F1 Score)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_precision_recall_scatter(perf_df: pd.DataFrame, output_path: Path):
    """Scatter plot of precision vs recall for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(perf_df["recall"], perf_df["precision"], s=100, zorder=5)
    for _, row in perf_df.iterrows():
        ax.annotate(row["model"], (row["recall"], row["precision"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_agreement_distribution(models: dict, gt: np.ndarray, output_path: Path):
    """Histogram of how many models agree on each pixel."""
    all_preds = np.stack(list(models.values()), axis=0)  # (n_models, n_pixels)
    agreement = all_preds.sum(axis=0)  # how many models predict positive

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(agreement, bins=range(len(models) + 2), align="left", edgecolor="black")
    ax.set_xlabel("Number of models predicting positive")
    ax.set_ylabel("Number of pixels")
    ax.set_title("Model Agreement Distribution")
    ax.set_xticks(range(len(models) + 1))
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze ensemble diversity metrics")
    parser.add_argument("--predictions_dir", type=str, default="predictions",
                        help="Directory containing model prediction subdirectories")
    parser.add_argument("--split", type=str, default="test",
                        help="Data split to analyze (test or val)")
    parser.add_argument("--output_dir", type=str, default="diversity_analysis",
                        help="Output directory for results")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Binarization threshold for predictions")
    args = parser.parse_args()

    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from {predictions_dir} (split={args.split}, threshold={args.threshold})...")
    models_binary, models_soft, gt = load_predictions_and_gt(predictions_dir, args.split, args.threshold)
    print(f"Loaded {len(models_binary)} models, {len(gt)} pixels per model")

    # Per-model performance
    print("Computing model performance...")
    perf_df = compute_model_performance(models_binary, gt)
    perf_df.to_csv(output_dir / "model_performance.csv", index=False)
    print(perf_df[["model", "f1", "precision", "recall"]].to_string(index=False))

    # Pairwise diversity matrices
    print("\nComputing pairwise diversity matrices...")
    names, matrices = compute_pairwise_matrices(models_binary, models_soft, gt)

    for metric_name, matrix in matrices.items():
        save_matrix_csv(names, matrix, output_dir / f"{metric_name}_matrix.csv")

    # Per-model diversity summary
    summary_df = compute_diversity_summary(names, matrices)
    summary_df.to_csv(output_dir / "diversity_summary.csv", index=False)
    print("\nDiversity summary (sorted by avg correlation, ascending = most diverse):")
    print(summary_df[["model", "avg_correlation", "avg_q_statistic", "avg_disagreement"]].to_string(index=False))

    # Plots
    print("\nGenerating plots...")
    plot_heatmap(names, matrices["correlation"], "Pairwise Correlation (Phi Coefficient)",
                 output_dir / "heatmap_correlation.png", vmin=0.5, vmax=1.0)
    plot_heatmap(names, matrices["q_statistic"], "Pairwise Q-Statistic",
                 output_dir / "heatmap_q_statistic.png", vmin=0.98, vmax=1.0)
    plot_heatmap(names, matrices["disagreement"], "Pairwise Disagreement",
                 output_dir / "heatmap_disagreement.png", vmin=0, cmap="YlOrRd")
    plot_heatmap(names, matrices["double_fault"], "Pairwise Double Fault",
                 output_dir / "heatmap_double_fault.png", vmin=0, cmap="YlOrRd")
    plot_model_performance(perf_df, output_dir / "model_performance.png")
    plot_precision_recall_scatter(perf_df, output_dir / "precision_recall_scatter.png")
    plot_agreement_distribution(models_binary, gt, output_dir / "agreement_distribution.png")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
