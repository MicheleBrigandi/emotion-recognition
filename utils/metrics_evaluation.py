# Michele Brigandì - 2156373

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from utils.logger import get_logger

logger = get_logger(__name__)

def save_evaluation_metrics(true_labels: np.ndarray, 
                            pred_labels: np.ndarray,
                            output_dir: str,
                            class_names: list[str] | None = None, 
                            file_format: str = "png"
):
    """
    Compute and save evaluation metrics for a classification model.

    Specifically:
        - Accuracy
        - Classification report (precision, recall, f1-score)
        - Confusion matrix (saved as a figure)

    Results are saved inside a timestamped subdirectory (e.g., results/2025-09-08_10-15-00)
    to avoid overwriting previous results.

    Args:
        true_labels (np.ndarray): Ground truth labels of shape (n_samples,), dtype int.
        pred_labels (np.ndarray): Predicted labels of shape (n_samples,), dtype int.
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

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap=plt.cm.get_cmap("Blues"), colorbar=True)
    plt.title("Confusion Matrix")
    plt.savefig(output / f"confusion_matrix.{file_format}")
    plt.close(fig)

    logger.info(f"Results saved to '{output_dir}'")
