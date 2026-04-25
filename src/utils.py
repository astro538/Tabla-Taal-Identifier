"""
Utility functions for visualization, metrics reporting, and results management.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import pandas as pd


# Style configuration
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "figure.dpi": 150,
    "font.size": 10,
    "font.family": "sans-serif",
})

COLORS = ["#58a6ff", "#f0883e", "#3fb950", "#d2a8ff",
          "#ff7b72", "#79c0ff", "#56d364", "#e3b341"]


def plot_training_curves(results_dict, task_name, save_path):
    """Plot training and validation loss/accuracy curves for all models.
    
    Args:
        results_dict: dict of {model_name: {history: {train_loss, val_loss, ...}}}
        task_name: 'Bol' or 'Taal' for the title
        save_path: path to save the plot
    """
    n_models = len(results_dict)
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 8))
    if n_models == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(f"{task_name} Classification — Training Curves",
                 fontsize=16, fontweight="bold", y=0.98)

    for idx, (model_name, result) in enumerate(results_dict.items()):
        hist = result["history"]
        epochs = range(1, len(hist["train_loss"]) + 1)

        # Loss plot
        ax = axes[0, idx]
        ax.plot(epochs, hist["train_loss"], color=COLORS[0], linewidth=1.5,
                label="Train", alpha=0.9)
        ax.plot(epochs, hist["val_loss"], color=COLORS[1], linewidth=1.5,
                label="Val", alpha=0.9)
        ax.set_title(model_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

        # Accuracy plot
        ax = axes[1, idx]
        ax.plot(epochs, hist["train_acc"], color=COLORS[2], linewidth=1.5,
                label="Train", alpha=0.9)
        ax.plot(epochs, hist["val_acc"], color=COLORS[3], linewidth=1.5,
                label="Val", alpha=0.9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved training curves → {save_path}")


def plot_confusion_matrices(results_dict, class_names, task_name, save_path):
    """Plot confusion matrices for all models in a single figure.
    
    Args:
        results_dict: dict of {model_name: {confusion_matrix: np.array}}
        class_names: list of class name strings
        task_name: 'Bol' or 'Taal'
        save_path: path to save the plot
    """
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    fig.suptitle(f"{task_name} Classification — Confusion Matrices",
                 fontsize=16, fontweight="bold", y=1.02)

    for idx, (model_name, result) in enumerate(results_dict.items()):
        ax = axes[idx]
        cm = result["confusion_matrix"]

        # Normalize confusion matrix
        cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, cbar=False, linewidths=0.5, linecolor="#30363d",
            annot_kws={"size": 8}
        )
        ax.set_title(f"{model_name}\nAcc: {result['accuracy']:.1%}",
                      fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.tick_params(axis="both", labelsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrices → {save_path}")


def plot_model_comparison(results_dict, task_name, save_path):
    """Plot bar chart comparing model performance metrics.
    
    Args:
        results_dict: dict of {model_name: {accuracy, precision, recall, f1_score}}
        task_name: 'Bol' or 'Taal'
        save_path: path to save the plot
    """
    models = list(results_dict.keys())
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(models))
    width = 0.18
    offsets = np.arange(len(metrics)) - (len(metrics) - 1) / 2

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results_dict[m].get(metric, 0) for m in models]
        bars = ax.bar(x + offsets[i] * width, values, width,
                      label=label, color=COLORS[i], alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=7,
                    color="#c9d1d9", fontweight="bold")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"{task_name} Classification — Model Comparison",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved model comparison → {save_path}")


def plot_final_comparison(bol_results, taal_results, save_path):
    """Plot combined comparison of all models on both tasks.
    
    Args:
        bol_results: dict of {model_name: {accuracy, f1_score, ...}}
        taal_results: dict of same structure
        save_path: path to save the plot
    """
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
                        f"{h:.1%}", ha="center", fontsize=7, color="#c9d1d9",
                        fontweight="bold")

        ax.set_title(f"{task_name} Classification", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9, rotation=15)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Tabla Taal & Bol Classification — Final Comparison",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved final comparison → {save_path}")


def plot_per_class_f1(results_dict, class_names, task_name, save_path):
    """Plot per-class F1 scores for each model as a heatmap."""
    models = list(results_dict.keys())
    data = []

    for model_name in models:
        report = results_dict[model_name].get("classification_report", {})
        row = []
        for cls in class_names:
            f1 = report.get(cls, {}).get("f1-score", 0)
            row.append(f1)
        data.append(row)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 1.2), len(models) * 1.2 + 1))
    sns.heatmap(
        data, annot=True, fmt=".2f", cmap="RdYlGn",
        xticklabels=class_names, yticklabels=models,
        ax=ax, linewidths=0.5, linecolor="#30363d",
        vmin=0, vmax=1, annot_kws={"size": 9}
    )
    ax.set_title(f"{task_name} — Per-Class F1 Scores",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Model", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved per-class F1 heatmap → {save_path}")


def print_results_table(results_dict, task_name):
    """Print a formatted results table to console."""
    print(f"\n{'=' * 75}")
    print(f"  {task_name} Classification Results")
    print(f"{'=' * 75}")
    print(f"  {'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
          f"{'F1 Score':>10} {'Epochs':>8}")
    print(f"  {'-' * 68}")

    best_acc = 0
    best_model = ""

    for model_name, result in results_dict.items():
        acc = result["accuracy"]
        if acc > best_acc:
            best_acc = acc
            best_model = model_name

        marker = " ★" if model_name == best_model and len(results_dict) > 1 else ""
        print(f"  {model_name:<20} {result['accuracy']:>9.2%} "
              f"{result['precision']:>9.2%} {result['recall']:>9.2%} "
              f"{result['f1_score']:>9.2%} {result['epochs_trained']:>7d}{marker}")

    print(f"{'=' * 75}")
    if len(results_dict) > 1:
        print(f"  ★ Best model: {best_model} ({best_acc:.2%} accuracy)")
    print()


def save_results_json(results_dict, save_path):
    """Save results to a JSON file (excluding non-serializable objects)."""
    serializable = {}
    for model_name, result in results_dict.items():
        entry = {
            "accuracy": float(result["accuracy"]),
            "precision": float(result["precision"]),
            "recall": float(result["recall"]),
            "f1_score": float(result["f1_score"]),
            "epochs_trained": int(result["epochs_trained"]),
            "confusion_matrix": result["confusion_matrix"].tolist(),
        }

        # Add per-class metrics
        report = result.get("classification_report", {})
        per_class = {}
        for cls_name, metrics in report.items():
            if isinstance(metrics, dict) and "f1-score" in metrics:
                per_class[cls_name] = {
                    k: float(v) for k, v in metrics.items()
                }
        entry["per_class"] = per_class
        serializable[model_name] = entry

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved results JSON → {save_path}")


def save_results_csv(results_dict, save_path):
    """Save comparison results as a CSV table."""
    rows = []
    for model_name, result in results_dict.items():
        rows.append({
            "Model": model_name,
            "Accuracy": f"{result['accuracy']:.4f}",
            "Precision": f"{result['precision']:.4f}",
            "Recall": f"{result['recall']:.4f}",
            "F1 Score": f"{result['f1_score']:.4f}",
            "Epochs": result["epochs_trained"],
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"  Saved results CSV → {save_path}")


def generate_all_plots(results_dict, class_names, task_name, output_dir):
    """Generate all plots for a given task.
    
    Args:
        results_dict: dict of model results
        class_names: list of class names
        task_name: 'Bol' or 'Taal'
        output_dir: directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    plot_training_curves(
        results_dict, task_name,
        os.path.join(output_dir, "training_curves.png")
    )
    plot_confusion_matrices(
        results_dict, class_names, task_name,
        os.path.join(output_dir, "confusion_matrices.png")
    )
    plot_model_comparison(
        results_dict, task_name,
        os.path.join(output_dir, "model_comparison.png")
    )
    plot_per_class_f1(
        results_dict, class_names, task_name,
        os.path.join(output_dir, "per_class_f1.png")
    )
    save_results_json(
        results_dict,
        os.path.join(output_dir, "results.json")
    )
    save_results_csv(
        results_dict,
        os.path.join(output_dir, "results.csv")
    )
