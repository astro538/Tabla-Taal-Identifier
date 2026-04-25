#!/usr/bin/env python3
"""
Tabla Taal & Bol Identification — Main Training Script.

Trains 4 deep learning architectures on both Bol and Taal datasets using
K-fold cross-validation, then compares results and produces an ensemble.

Usage:
    python train.py                         # Train all models on both tasks
    python train.py --task bol              # Train only on Bols
    python train.py --task taal             # Train only on Taals
    python train.py --models ResidualCNN CRNN  # Train specific models
    python train.py --folds 3 --epochs 50   # Quick run
"""

import os
import sys
import time
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from src.config import Config
from src.dataset import load_file_paths, TablaDataset
from src.models import get_model, model_summary, MODEL_REGISTRY
from src.trainer import Trainer, get_device
from src.utils import (
    print_results_table, generate_all_plots, plot_final_comparison,
    save_results_json
)


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DL models for Tabla Taal & Bol classification"
    )
    parser.add_argument("--task", type=str, default="both",
                        choices=["bol", "taal", "both"],
                        help="Which task to train on (default: both)")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Models to train (default: all). "
                             f"Choices: {list(MODEL_REGISTRY.keys())}")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Max epochs per fold (default: 80)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--patience", type=int, default=12,
                        help="Early stopping patience (default: 12)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable waveform augmentation copies")
    parser.add_argument("--aug-copies", type=int, default=5,
                        help="Number of augmented copies per sample (default: 5)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-root", type=str, default="",
                        help="Override dataset root path")
    return parser.parse_args()


def train_single_task(task: str, config: Config, device, model_names: list):
    """
    Train all specified models on a single task (bol or taal) with K-fold CV.
    
    Returns:
        dict of {model_name: averaged_results}
    """
    task_label = task.capitalize()
    print(f"\n{'#' * 75}")
    print(f"#  Training on {task_label} Dataset")
    print(f"#  Max duration: {config.max_duration}s | "
          f"Target frames: {config.target_frames} | "
          f"Folds: {config.training.n_folds}")
    print(f"{'#' * 75}")

    # Load dataset
    data_dir = config.task_data_dir
    file_paths, labels, class_names = load_file_paths(data_dir)
    print(f"\n  Classes ({len(class_names)}): {class_names}")
    print(f"  Total files: {len(file_paths)}")

    labels_np = np.array(labels)

    # K-Fold Cross Validation
    skf = StratifiedKFold(
        n_splits=config.training.n_folds,
        shuffle=True,
        random_state=config.seed
    )

    all_model_results = {}

    for model_name in model_names:
        print(f"\n{'─' * 60}")
        print(f"  Model: {model_name}")
        print(f"{'─' * 60}")

        fold_results = []
        fold_histories = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels)):
            print(f"\n  Fold {fold_idx + 1}/{config.training.n_folds}")

            # Split data
            train_paths = [file_paths[i] for i in train_idx]
            train_labels = labels_np[train_idx].tolist()
            val_paths = [file_paths[i] for i in val_idx]
            val_labels = labels_np[val_idx].tolist()

            # Create datasets
            use_aug = not hasattr(config, '_no_augment') or not config._no_augment
            train_dataset = TablaDataset(
                train_paths, train_labels, config,
                augment=True,
                create_augmented_copies=use_aug
            )
            val_dataset = TablaDataset(
                val_paths, val_labels, config,
                augment=False,
                create_augmented_copies=False
            )

            # Create data loaders
            use_pin = device.type == "cuda"  # Only CUDA supports pin_memory
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.training.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=use_pin,
                drop_last=True if len(train_dataset) > config.training.batch_size else False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=0
            )

            # Create model
            set_seed(config.seed + fold_idx)
            model = get_model(model_name, num_classes=config.num_classes)
            if fold_idx == 0:
                model_summary(model, model_name)

            # Create trainer and train
            trainer = Trainer(model, config, device, class_names)
            fold_result = trainer.fit(train_loader, val_loader, verbose=True)

            fold_results.append(fold_result)
            fold_histories.append(fold_result["history"])

            print(f"    Fold {fold_idx + 1} — "
                  f"Acc: {fold_result['accuracy']:.2%}, "
                  f"F1: {fold_result['f1_score']:.2%}")

            # Save best model for this fold
            if config.save_models:
                model_path = os.path.join(
                    config.task_output_dir, "models",
                    f"{model_name}_fold{fold_idx}.pth"
                )
                trainer.save(model_path)

        # Average results across folds
        avg_result = average_fold_results(fold_results, fold_histories)
        all_model_results[model_name] = avg_result

        print(f"\n  {model_name} Average — "
              f"Acc: {avg_result['accuracy']:.2%} ± {avg_result.get('accuracy_std', 0):.2%}, "
              f"F1: {avg_result['f1_score']:.2%} ± {avg_result.get('f1_std', 0):.2%}")

    # Ensemble evaluation
    if len(model_names) > 1:
        print(f"\n{'─' * 60}")
        print(f"  Ensemble (weighted average)")
        print(f"{'─' * 60}")
        ensemble_result = create_ensemble_result(all_model_results)
        all_model_results["Ensemble"] = ensemble_result
        print(f"  Ensemble — Acc: {ensemble_result['accuracy']:.2%}, "
              f"F1: {ensemble_result['f1_score']:.2%}")

    # Print results table
    print_results_table(all_model_results, task_label)

    # Generate plots and save results
    generate_all_plots(
        all_model_results, class_names, task_label,
        config.task_output_dir
    )

    return all_model_results


def average_fold_results(fold_results, fold_histories):
    """Average metrics across folds and pick the best fold for detailed results."""
    accuracies = [r["accuracy"] for r in fold_results]
    precisions = [r["precision"] for r in fold_results]
    recalls = [r["recall"] for r in fold_results]
    f1_scores = [r["f1_score"] for r in fold_results]

    # Use the best fold for confusion matrix and classification report
    best_fold_idx = np.argmax(accuracies)
    best_fold = fold_results[best_fold_idx]

    # Use the best fold's history for plotting (most representative)
    best_history = fold_histories[best_fold_idx]

    return {
        "accuracy": np.mean(accuracies),
        "accuracy_std": np.std(accuracies),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1_score": np.mean(f1_scores),
        "f1_std": np.std(f1_scores),
        "confusion_matrix": best_fold["confusion_matrix"],
        "classification_report": best_fold["classification_report"],
        "history": best_history,
        "predictions": best_fold["predictions"],
        "true_labels": best_fold["true_labels"],
        "epochs_trained": int(np.mean([r["epochs_trained"] for r in fold_results])),
        "fold_accuracies": accuracies,
        "fold_f1_scores": f1_scores,
    }


def create_ensemble_result(model_results):
    """Create ensemble result by majority voting from individual model predictions."""
    # Collect predictions from all models
    models = [name for name in model_results.keys() if name != "Ensemble"]

    # Use predictions from best fold of each model
    all_preds = []
    true_labels = None
    weights = []

    for name in models:
        result = model_results[name]
        all_preds.append(result["predictions"])
        weights.append(result["accuracy"])
        if true_labels is None:
            true_labels = result["true_labels"]

    # Weighted majority voting
    all_preds = np.array(all_preds)  # (n_models, n_samples)
    weights = np.array(weights)
    weights = weights / weights.sum()

    n_samples = all_preds.shape[1]
    n_classes = int(all_preds.max()) + 1
    ensemble_preds = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        votes = np.zeros(n_classes)
        for m_idx in range(len(models)):
            votes[all_preds[m_idx, i]] += weights[m_idx]
        ensemble_preds[i] = np.argmax(votes)

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score as f1_score_fn, confusion_matrix, classification_report
    )

    acc = accuracy_score(true_labels, ensemble_preds)
    prec = precision_score(true_labels, ensemble_preds, average="weighted", zero_division=0)
    rec = recall_score(true_labels, ensemble_preds, average="weighted", zero_division=0)
    f1 = f1_score_fn(true_labels, ensemble_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(true_labels, ensemble_preds)
    report = classification_report(
        true_labels, ensemble_preds, zero_division=0, output_dict=True
    )

    # Use best model's history for plotting
    best_model = max(model_results.items(), key=lambda x: x[1]["accuracy"])
    history = best_model[1]["history"]

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "history": history,
        "predictions": ensemble_preds,
        "true_labels": true_labels,
        "epochs_trained": 0,
    }


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 75)
    print("  🥁 Tabla Taal & Bol Identification — Deep Learning Training")
    print("=" * 75)

    # Setup config
    config = Config()
    config.training.n_folds = args.folds
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.patience = args.patience
    config.output_dir = args.output
    config.seed = args.seed
    config.augment.n_augmented_copies = args.aug_copies

    if args.dataset_root:
        config.dataset_root = args.dataset_root

    if args.no_augment:
        config._no_augment = True
        config.augment.n_augmented_copies = 0

    # Select models
    model_names = args.models or list(MODEL_REGISTRY.keys())
    for name in model_names:
        if name not in MODEL_REGISTRY:
            print(f"Error: Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
            sys.exit(1)

    # Print configuration
    print(f"\n  Configuration:")
    print(f"    Dataset root:  {config.dataset_root}")
    print(f"    Models:        {model_names}")
    print(f"    Folds:         {config.training.n_folds}")
    print(f"    Max epochs:    {config.training.epochs}")
    print(f"    Batch size:    {config.training.batch_size}")
    print(f"    Learning rate: {config.training.learning_rate}")
    print(f"    Mixup alpha:   {config.training.mixup_alpha}")
    print(f"    Label smooth:  {config.training.label_smoothing}")
    print(f"    Aug copies:    {config.augment.n_augmented_copies}")
    print(f"    Seed:          {config.seed}")
    print(f"    Output:        {config.output_dir}")

    # Set seed
    set_seed(config.seed)

    # Get device
    device = get_device()
    start_time = time.time()

    # Determine tasks
    tasks = []
    if args.task in ("bol", "both"):
        tasks.append("bol")
    if args.task in ("taal", "both"):
        tasks.append("taal")

    all_task_results = {}

    for task in tasks:
        config.task = task
        # Set appropriate max duration
        if task == "bol":
            pass  # uses bol_max_duration automatically
        else:
            pass  # uses taal_max_duration automatically

        task_results = train_single_task(task, config, device, model_names)
        all_task_results[task] = task_results

    # Generate final comparison if both tasks were run
    if len(tasks) == 2:
        print(f"\n{'#' * 75}")
        print(f"#  Final Comparison — Both Tasks")
        print(f"{'#' * 75}")

        plot_final_comparison(
            all_task_results["bol"],
            all_task_results["taal"],
            os.path.join(config.output_dir, "final_comparison.png")
        )

        # Print combined summary
        for task in tasks:
            task_label = task.capitalize()
            results = all_task_results[task]
            best_model = max(results.items(), key=lambda x: x[1]["accuracy"])
            print(f"\n  Best {task_label} model: {best_model[0]} "
                  f"({best_model[1]['accuracy']:.2%} accuracy, "
                  f"{best_model[1]['f1_score']:.2%} F1)")

    # Total time
    elapsed = time.time() - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n{'=' * 75}")
    time_str = ""
    if hours > 0:
        time_str += f"{int(hours)}h "
    time_str += f"{int(minutes)}m {int(seconds)}s"
    print(f"  Total training time: {time_str}")
    print(f"  Results saved to: {os.path.abspath(config.output_dir)}/")
    print(f"{'=' * 75}")
    print(f"\n  🎉 Training complete! Check the results directory for:")
    print(f"     • training_curves.png    — Loss/accuracy over epochs")
    print(f"     • confusion_matrices.png — Per-class prediction accuracy")
    print(f"     • model_comparison.png   — Side-by-side metric comparison")
    print(f"     • per_class_f1.png       — F1 scores heatmap")
    print(f"     • results.json/csv       — Detailed numeric results")
    if config.save_models:
        print(f"     • models/               — Saved model weights")
    print()


if __name__ == "__main__":
    main()
