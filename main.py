import cv2
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from keras.models import load_model
from pathlib import Path

from config.config import Config
from models.custom_layers.attention_blocks import SEBlock, CAMBlock
from preprocessing.image_preprocessing import preprocess_images
from utils.logger import get_logger
from utils.metrics_evaluation import save_evaluation_metrics
from utils.visualization import draw_predictions_on_faces

config = Config()
logger = get_logger(__name__)

def main():
    logger.info("Checking if preprocessing is needed...")
    preprocessed_folder = Path(config.preprocessed_images_folder)
    preprocessed_csv_path = Path(config.preprocessed_csv)

    if not preprocessed_folder.is_dir() or not preprocessed_csv_path.exists():
        logger.info("Preprocessed data not found. Running preprocessing...")
        preprocess_images(
            raw_images_dir=config.raw_images_folder,
            detections_csv_path=config.detections_csv,
            preprocessed_images_dir=config.preprocessed_images_folder,
            preprocessed_csv_path=config.preprocessed_csv,
            target_size=config.image_size[0]
        )

    logger.info("Checking if model is present...")
    model_path = Path(config.model_path)

    if not model_path.exists():
        logger.info("Model not found. Downloading...")
        model_file = hf_hub_download(
            repo_id="michelebrigandi/emotion-recognition-cnn-keras",
            filename="emotion-recognition-cnn.keras"
        )

        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(Path(model_file).read_bytes())
        logger.info(f"Model ready at: {model_path}")

    logger.info("Loading model...")
    model = load_model(config.model_path, custom_objects={"SEBlock": SEBlock, "CAMBlock": CAMBlock})

    logger.info("Loading preprocessed images...")
    images, image_files = load_preprocessed_images(config.preprocessed_images_folder)
    logger.info(f"Loaded {len(images)} images.")

    logger.info("Loading labels...")
    valid_idx, y_true = load_labels(image_files, config.preprocessed_csv)
    images = images[valid_idx]
    logger.info(f"Using {len(y_true)} valid labels for evaluation.")

    logger.info("Running predictions...")
    preds = model.predict(images, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    # Save predictions CSV
    df = pd.read_csv(preprocessed_csv_path)
    df["prediction"] = y_pred
    df.to_csv(config.predictions_csv, index=False)
    logger.info(f"Predictions saved in '{config.predictions_csv}'.")

    # Save evaluation metrics
    class_names = [config.emotion_labels[i] for i in sorted(config.emotion_labels)]
    save_evaluation_metrics(y_true, y_pred, config.result_metrics_folder, class_names)

    # Draw predictions on images
    draw_predictions_on_faces(
        predictions_file_path=config.predictions_csv,
        original_images_dir=config.raw_images_folder,
        output_dir=config.result_images_folder,
        emotion_labels=config.emotion_labels
    )

    logger.info("Pipeline completed successfully.")


def load_preprocessed_images(preprocessed_images_dir: str) -> tuple[np.ndarray, list[Path]]:
    """
    Load preprocessed grayscale images and return them as a numpy array along with file paths.

    Args:
        preprocessed_images_dir (str): Path to directory containing preprocessed images.

    Returns:
        tuple[np.ndarray, list[Path]]: Array of images (N, H, W, 1) normalized to [0,1]
                                       and list of corresponding file paths.
    """
    images, image_files = [], sorted(Path(preprocessed_images_dir).glob("*.*"))
    for file in image_files:
        img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"Cannot load image: {file}")
            continue

        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        images.append(img)

    return np.array(images), image_files


def load_labels(image_files: list[Path], labels_csv: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load labels corresponding to preprocessed images, filtering out missing or invalid entries.

    Args:
        image_files (list[Path]): List of preprocessed image files.
        labels_csv (str): CSV path containing labels with column 'new_filename' and 'label'.

    Returns:
        tuple[np.ndarray, np.ndarray]: Boolean mask of valid images and array of valid labels.
    """
    labels_df = pd.read_csv(labels_csv)
    labels = []
    for file in image_files:
        row = labels_df[labels_df['new_image'] == file.name]
        if len(row) == 1:
            labels.append(row['label'].values[0])
        else:
            logger.warning(f"Missing or duplicate label for: {file.name}")
            labels.append(-1)

    labels = np.array(labels).astype(int)
    valid_idx = labels >= 0

    return valid_idx, labels[valid_idx]


if __name__ == "__main__":
    main()
