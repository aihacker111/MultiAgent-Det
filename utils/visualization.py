from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from utils.metrics import AnalysisReport, EpochMetrics


def plot_training_curves(history: list[EpochMetrics], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [m.epoch for m in history]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

    pairs = [
        ("box_loss", [m.train_box_loss for m in history], [m.val_box_loss for m in history]),
        ("cls_loss", [m.train_cls_loss for m in history], [m.val_cls_loss for m in history]),
        ("dfl_loss", [m.train_dfl_loss for m in history], [m.val_dfl_loss for m in history]),
    ]
    for ax, (name, train_vals, val_vals) in zip(axes[0], pairs):
        ax.plot(epochs, train_vals, label="train", color="#2196F3")
        ax.plot(epochs, val_vals, label="val", color="#F44336", linestyle="--")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[1][0].plot(epochs, [m.map50 for m in history], color="#4CAF50")
    axes[1][0].set_title("mAP50")
    axes[1][0].grid(alpha=0.3)

    axes[1][1].plot(epochs, [m.map50_95 for m in history], color="#9C27B0")
    axes[1][1].set_title("mAP50-95")
    axes[1][1].grid(alpha=0.3)

    axes[1][2].semilogy(epochs, [m.lr for m in history], color="#FF9800")
    axes[1][2].set_title("Learning Rate")
    axes[1][2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    save_path: Path,
    normalize: bool = True,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        matrix = matrix.astype(float) / row_sums

    fig_size = max(8, len(class_names) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        matrix,
        annot=len(class_names) <= 20,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""), fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_ap(report: AnalysisReport, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not report.class_metrics:
        return

    sorted_metrics = sorted(report.class_metrics, key=lambda m: m.ap50)
    classes = [m.class_name for m in sorted_metrics]
    ap50_vals = [m.ap50 for m in sorted_metrics]
    ap50_95_vals = [m.ap50_95 for m in sorted_metrics]

    y = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(10, max(5, len(classes) * 0.35)))
    bars = ax.barh(y - 0.2, ap50_vals, height=0.35, label="AP50", color="#2196F3", alpha=0.85)
    ax.barh(y + 0.2, ap50_95_vals, height=0.35, label="AP50-95", color="#4CAF50", alpha=0.85)
    ax.axvline(x=0.3, color="#F44336", linestyle="--", alpha=0.6, label="Weak threshold (0.3)")
    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Average Precision")
    ax.set_title("Per-class AP Breakdown")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_size_breakdown(report: AnalysisReport, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    sm = report.size_metrics
    labels = [f"Small\n(n={sm.small_count})", f"Medium\n(n={sm.medium_count})", f"Large\n(n={sm.large_count})"]
    values = [sm.small_map50, sm.medium_map50, sm.large_map50]
    colors = ["#FF5722", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)
    ax.axhline(y=0.3, color="#F44336", linestyle="--", alpha=0.6, label="Weak threshold")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("mAP50")
    ax.set_title("mAP50 by Object Size")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
