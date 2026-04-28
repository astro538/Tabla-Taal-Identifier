"""
Regenerate all result plots with white backgrounds for report/thesis use.
Reads data from results JSON + model checkpoints, outputs to results_white/.
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─── White-background style ──────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cccccc",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "grid.color": "#e0e0e0",
    "figure.dpi": 200,
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea",
          "#ea580c", "#0891b2", "#65a30d", "#ca8a04"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_DIR = os.path.join(BASE_DIR, "results_white")


# ─── Load training history from checkpoints ──────────────────────────
def load_histories(task_dir):
    """Load training histories from .pth checkpoints."""
    models_dir = os.path.join(task_dir, "models")
    histories = {}
    if not os.path.exists(models_dir):
        return histories

    model_names = set()
    for f in os.listdir(models_dir):
        if f.endswith(".pth"):
            name = f.rsplit("_fold", 1)[0]
            model_names.add(name)

    for name in sorted(model_names):
        # Load fold 0 as representative
        fold_path = os.path.join(models_dir, f"{name}_fold0.pth")
        if os.path.exists(fold_path):
            try:
                ckpt = torch.load(fold_path, map_location="cpu", weights_only=False)
                if "history" in ckpt:
                    histories[name] = ckpt["history"]
                    print(f"  ✓ Loaded history for {name} ({len(ckpt['history']['train_loss'])} epochs)")
            except Exception as e:
                print(f"  ✗ Failed to load {name}: {e}")
    return histories


# ─── Load results JSON ───────────────────────────────────────────────
def load_results_json(task_dir):
    """Load results.json for a task."""
    json_path = os.path.join(task_dir, "results.json")
    with open(json_path, "r") as f:
        return json.load(f)


# ─── Plot: Training Curves ───────────────────────────────────────────
def plot_training_curves(histories, task_name, save_path):
    """Plot training and validation loss/accuracy curves."""
    model_names = list(histories.keys())
    n = len(model_names)
    if n == 0:
        print(f"  ⚠ No history data for {task_name}, skipping training curves")
        return

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(f"{task_name} Classification — Training Curves",
                 fontsize=16, fontweight="bold", y=0.98, color="#111111")

    for idx, name in enumerate(model_names):
        hist = histories[name]
        epochs = range(1, len(hist["train_loss"]) + 1)

        # Loss subplot
        ax = axes[0, idx]
        ax.plot(epochs, hist["train_loss"], color=COLORS[0], linewidth=1.8,
                label="Train", alpha=0.9)
        ax.plot(epochs, hist["val_loss"], color=COLORS[1], linewidth=1.8,
                label="Val", alpha=0.9)
        ax.set_title(name, fontsize=12, fontweight="bold", color="#111111")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9,
                  edgecolor="#cccccc")
        ax.grid(True, alpha=0.3)

        # Accuracy subplot
        ax = axes[1, idx]
        ax.plot(epochs, hist["train_acc"], color=COLORS[2], linewidth=1.8,
                label="Train", alpha=0.9)
        ax.plot(epochs, hist["val_acc"], color=COLORS[3], linewidth=1.8,
                label="Val", alpha=0.9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9,
                  edgecolor="#cccccc")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Saved {save_path}")


# ─── Plot: Confusion Matrices (2-row layout: 3 top, 2 bottom centered) ──
def plot_confusion_matrices(results, class_names, task_name, save_path):
    """Plot confusion matrices in a 2-row grid: 3 on top, 2 centered below."""
    from matplotlib.gridspec import GridSpec

    model_names = list(results.keys())
    n = len(model_names)

    # Determine grid layout
    if n <= 3:
        # Single row for 3 or fewer
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
        if n == 1:
            axes = [axes]
        axes_list = list(axes)
    else:
        # 2-row layout: first 3 on top, remaining centered on bottom
        top_count = 3
        bot_count = n - top_count

        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 6, figure=fig, hspace=0.35, wspace=0.4)

        axes_list = []

        # Top row: 3 matrices evenly across 6 columns
        for i in range(top_count):
            ax = fig.add_subplot(gs[0, i * 2:(i + 1) * 2])
            axes_list.append(ax)

        # Bottom row: center the remaining matrices
        # For 2 matrices in a 6-col grid, place at columns 1-2 and 3-4
        offset = (6 - bot_count * 2) // 2
        for i in range(bot_count):
            col_start = offset + i * 2
            ax = fig.add_subplot(gs[1, col_start:col_start + 2])
            axes_list.append(ax)

    fig.suptitle(f"{task_name} Classification — Confusion Matrices",
                 fontsize=18, fontweight="bold", y=0.98, color="#111111")

    for idx, name in enumerate(model_names):
        ax = axes_list[idx]
        cm = np.array(results[name]["confusion_matrix"])
        cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, cbar=False, linewidths=0.5, linecolor="#cccccc",
            annot_kws={"size": 9, "color": "#222222"}
        )
        acc = results[name]["accuracy"]
        ax.set_title(f"{name}\nAcc: {acc:.1%}",
                     fontsize=12, fontweight="bold", color="#111111")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.tick_params(axis="both", labelsize=8)

    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Saved {save_path}")


# ─── Plot: Model Comparison Bar Chart ────────────────────────────────
def plot_model_comparison(results, task_name, save_path):
    """Plot bar chart comparing model performance metrics."""
    models = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(models))
    width = 0.18
    offsets = np.arange(len(metrics)) - (len(metrics) - 1) / 2

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[m].get(metric, 0) for m in models]
        bars = ax.bar(x + offsets[i] * width, values, width,
                      label=label, color=COLORS[i], alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=7,
                    color="#222222", fontweight="bold")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"{task_name} Classification — Model Comparison",
                 fontsize=14, fontweight="bold", color="#111111")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9, edgecolor="#cccccc")
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Saved {save_path}")


# ─── Plot: Per-Class F1 Heatmap ──────────────────────────────────────
def plot_per_class_f1(results, class_names, task_name, save_path):
    """Plot per-class F1 scores as a heatmap."""
    models = list(results.keys())
    data = []

    for name in models:
        per_class = results[name].get("per_class", {})
        row = []
        for cls in class_names:
            f1 = per_class.get(cls, {}).get("f1-score", 0)
            row.append(f1)
        data.append(row)

    data = np.array(data)

    fig, ax = plt.subplots(
        figsize=(max(8, len(class_names) * 1.2), len(models) * 1.2 + 1)
    )
    sns.heatmap(
        data, annot=True, fmt=".2f", cmap="RdYlGn",
        xticklabels=class_names, yticklabels=models,
        ax=ax, linewidths=0.5, linecolor="#cccccc",
        vmin=0, vmax=1, annot_kws={"size": 9, "color": "#222222"}
    )
    ax.set_title(f"{task_name} — Per-Class F1 Scores",
                 fontsize=13, fontweight="bold", color="#111111")
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Saved {save_path}")


# ─── Plot: Final Comparison (Bol + Taal side-by-side) ────────────────
def plot_final_comparison(bol_results, taal_results, save_path):
    """Plot combined Bol + Taal comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (task_name, results) in zip(axes, [("Bol", bol_results), ("Taal", taal_results)]):
        models = list(results.keys())
        acc = [results[m]["accuracy"] for m in models]
        f1 = [results[m]["f1_score"] for m in models]

        x = np.arange(len(models))
        width = 0.3

        bars1 = ax.bar(x - width / 2, acc, width, label="Accuracy",
                       color=COLORS[0], alpha=0.85, edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x + width / 2, f1, width, label="F1 Score",
                       color=COLORS[2], alpha=0.85, edgecolor="white", linewidth=0.5)

        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.1%}", ha="center", fontsize=7, color="#222222",
                        fontweight="bold")

        ax.set_title(f"{task_name} Classification", fontsize=13,
                     fontweight="bold", color="#111111")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9, rotation=15)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", fontsize=11)
        ax.legend(fontsize=9, framealpha=0.9, edgecolor="#cccccc")
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Tabla Taal & Bol Classification — Final Comparison",
                 fontsize=15, fontweight="bold", y=1.02, color="#111111")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓ Saved {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Regenerating plots with WHITE background for report")
    print("=" * 60)

    # Load data
    bol_dir = os.path.join(RESULTS_DIR, "bol")
    taal_dir = os.path.join(RESULTS_DIR, "taal")

    print("\n📂 Loading Bol results...")
    bol_results = load_results_json(bol_dir)
    bol_histories = load_histories(bol_dir)

    print("\n📂 Loading Taal results...")
    taal_results = load_results_json(taal_dir)
    taal_histories = load_histories(taal_dir)

    # Determine class names from per_class keys (exclude avg entries)
    bol_classes = [k for k in list(bol_results.values())[0].get("per_class", {}).keys()
                   if "avg" not in k]
    taal_classes = [k for k in list(taal_results.values())[0].get("per_class", {}).keys()
                    if "avg" not in k]

    print(f"\n  Bol classes: {bol_classes}")
    print(f"  Taal classes: {taal_classes}")

    # Create output directories
    bol_out = os.path.join(OUTPUT_DIR, "bol")
    taal_out = os.path.join(OUTPUT_DIR, "taal")
    os.makedirs(bol_out, exist_ok=True)
    os.makedirs(taal_out, exist_ok=True)

    # ─── Generate Bol plots ──────────────────────────────────────
    print("\n🎨 Generating Bol plots...")

    # Merge histories into results for training curves
    bol_with_hist = {}
    for name in bol_results:
        if name in bol_histories:
            bol_with_hist[name] = bol_histories[name]
    # Also try "Ensemble" which has epochs_trained=0, create a dummy if needed
    # Ensemble usually doesn't have a checkpoint

    plot_training_curves(
        bol_with_hist, "Bol",
        os.path.join(bol_out, "training_curves.png")
    )
    plot_confusion_matrices(
        bol_results, bol_classes, "Bol",
        os.path.join(bol_out, "confusion_matrices.png")
    )
    plot_model_comparison(
        bol_results, "Bol",
        os.path.join(bol_out, "model_comparison.png")
    )
    plot_per_class_f1(
        bol_results, bol_classes, "Bol",
        os.path.join(bol_out, "per_class_f1.png")
    )

    # ─── Generate Taal plots ─────────────────────────────────────
    print("\n🎨 Generating Taal plots...")

    taal_with_hist = {}
    for name in taal_results:
        if name in taal_histories:
            taal_with_hist[name] = taal_histories[name]

    plot_training_curves(
        taal_with_hist, "Taal",
        os.path.join(taal_out, "training_curves.png")
    )
    plot_confusion_matrices(
        taal_results, taal_classes, "Taal",
        os.path.join(taal_out, "confusion_matrices.png")
    )
    plot_model_comparison(
        taal_results, "Taal",
        os.path.join(taal_out, "model_comparison.png")
    )
    plot_per_class_f1(
        taal_results, taal_classes, "Taal",
        os.path.join(taal_out, "per_class_f1.png")
    )

    # ─── Final comparison ────────────────────────────────────────
    print("\n🎨 Generating final comparison...")
    plot_final_comparison(
        bol_results, taal_results,
        os.path.join(OUTPUT_DIR, "final_comparison.png")
    )

    print("\n" + "=" * 60)
    print(f"  ✅ All plots saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
