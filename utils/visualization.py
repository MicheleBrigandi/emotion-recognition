import cv2
import pandas as pd
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

def draw_predictions_on_faces(predictions_file_path: str, original_images_dir: str, output_dir: str, emotion_labels: dict[int, str]):
    """
    Draw predicted emotion labels and bounding boxes on images and save annotated images.

    Expects a CSV file containing predictions with the following columns:
        - original_image: file name of the original image
        - x, y, w, h: bounding box coordinates
        - prediction: integer class label of predicted emotion
        - label (optional): integer class label of true emotion, used to colour boxes

    Args:
        predictions_file_path (str): Path to the CSV file with predictions.
        original_images_dir (str): Directory containing the original images.
        output_dir (str): Directory where annotated images will be saved.
        emotion_labels (dict[int, str]): Mapping from integer class labels to emotion names.

    Raises:
        ValueError: If the CSV file is not in CSV format.
        FileNotFoundError: If the CSV file does not exist.
        RuntimeError: If the CSV file is empty.
    """
    predictions = load_predictions(predictions_file_path)
    images_dir = Path(original_images_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Group predictions by original file name
    grouped = predictions.groupby("original_image")

    for image_name, group in grouped:
        image_path = images_dir / str(image_name)
        if not image_path.is_file():
            logger.warning(f"Original image not found: {image_path}. Skipping.")
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Cannot load image: {image_path}. Skipping.")
            continue

        for _, row in group.iterrows():
            draw_single_bounding_box(img, row, emotion_labels)

        # Save the annotated image
        output_path = output / str(image_name)
        cv2.imwrite(str(output_path), img)

    logger.info(f"Output images saved to: {output_dir}")


def load_predictions(predictions_file_path: str) -> pd.DataFrame:
    """
    Load predictions from a CSV file.

    Raises:
        ValueError: If the file is not CSV.
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the CSV is empty.
    """
    predictions_file = Path(predictions_file_path)
    if predictions_file.suffix.lower() != ".csv":
        logger.error("The predictions file has to be in CSV format.")
        raise ValueError("The predictions file has to be in CSV format.")
    
    if not predictions_file.is_file():
        logger.error(f"Predictions file not found: {predictions_file_path}")
        raise FileNotFoundError(f"Predictions file not found: {predictions_file_path}")
    
    logger.info("Loading predictions data...")

    predictions = pd.read_csv(predictions_file)
    if predictions.empty:
        logger.error(f"Predictions file not found or empty: {predictions_file_path}")
        raise RuntimeError(f"Predictions file is empty: {predictions_file_path}")

    predictions["original_image"] = predictions["original_image"].astype(str)
    predictions["label"] = predictions["label"].astype(int)
    predictions["prediction"] = predictions["prediction"].astype(int)

    logger.info(f"{len(predictions)} predictions loaded.")

    return predictions


def draw_single_bounding_box(img, row, emotion_labels: dict[int, str]):
    """
    Draw a bounding box and emotion label on a single face in the image.

    Dynamic thickness and font size are scaled according to face size.
    Box is green if prediction matches label (if available), otherwise red.

    Args:
        img: The image array (OpenCV BGR) to draw on.
        row: A pandas Series representing one row from the predictions CSV.
        emotion_labels: Mapping from integer class labels to emotion names.
    """
    x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
    pred_label = row.get("prediction")
    true_label = row.get("label")
    label_name = emotion_labels.get(pred_label, str(pred_label))

    # Determine color: green if correct, red if wrong
    if true_label is not None:
        color = (0, 255, 0) if pred_label == true_label else (0, 0, 255)
    else:
        color = (0, 255, 0)

    # Dynamic scaling based on face size
    box_size = max(w, h)
    thickness = max(1, box_size // 100)
    font_scale = max(0.4, box_size / 200.0)
    text_thickness = max(1, thickness)

    # Draw rectangle and label
    cv2.rectangle(img, (x, y), (x + w, y + h), color=color, thickness=thickness)
    cv2.putText(
        img, 
        label_name, 
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, 
        color, 
        text_thickness
    )
