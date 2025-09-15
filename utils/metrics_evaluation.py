# Michele Brigandì - 2156373

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from utils.logger import get_logger

logger = get_logger(__name__)

def save_evaluation_metrics(true_labels: np.ndarray, 
                            pred_labels: np.ndarray,
                            ground_truth_csv: str,
                            output_dir: str,
                            class_names: list[str] | None = None, 
                            file_format: str = "png"
):
    """
    Compute and save evaluation metrics for a classification model.

    Specifically:
        - Accuracy
        - Classification report (precision, recall, f1-score)
        - End-to-end detection and classification success rate
        - Confusion matrix (saved as a figure)

    Results are saved inside a timestamped subdirectory (e.g., results/2025-09-08_10-15-00)
    to avoid overwriting previous results.

    Args:
        true_labels (np.ndarray): Ground truth labels of shape (n_samples,), dtype int.
        pred_labels (np.ndarray): Predicted labels of shape (n_samples,), dtype int.
        ground_truth_csv (str): Path to ground truth CSV file for counting total faces.
        output_dir (str): Base directory to save results.
        class_names (list[str], optional): List of class names ordered by index. Defaults to None.
        file_format (str, optional): Format to save confusion matrix figure. Defaults to "png".

    Notes:
        - Logs accuracy and classification report.
        - Saves classification report as a text file.
        - Confusion matrix figure is saved using matplotlib.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output = Path(output_dir) / timestamp
    output.mkdir(parents=True, exist_ok=True)

    # Accuracy
    acc = accuracy_score(true_labels, pred_labels)
    logger.info(f"Accuracy: {acc:.4f}")

    # Classification report
    report = classification_report(true_labels, pred_labels, target_names=class_names, zero_division=0)
    logger.info("\n" + str(report))
    (output / "classification_report.txt").write_text(str(report))

    # End-to-end detection + classification metric
    if ground_truth_csv is not None and Path(ground_truth_csv).exists():
        total_faces = len(pd.read_csv(ground_truth_csv))
        correct_faces = int(np.sum(true_labels == pred_labels))
        end_to_end_rate = correct_faces / total_faces if total_faces > 0 else 0.0

        logger.info(f"Detected faces percentage: {(len(true_labels) / total_faces):.4f} "
                    f"End-to-end success rate: {end_to_end_rate:.4f} "
                    f"({correct_faces}/{total_faces} faces)")

        # Save in the same report file
        with open(output / "classification_report.txt", "a") as f:
            f.write("\n")
            f.write(f"End-to-end success rate: {end_to_end_rate:.4f} "
                    f"({correct_faces}/{total_faces} faces)\n")

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap=plt.cm.get_cmap("Blues"), colorbar=True)
    plt.title("Confusion Matrix")
    plt.savefig(output / f"confusion_matrix.{file_format}")
    plt.close(fig)

    logger.info(f"Results saved to '{output_dir}'")
