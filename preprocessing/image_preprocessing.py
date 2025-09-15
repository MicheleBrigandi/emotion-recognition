# Michele Brigandì - 2156373

import cv2
import pandas as pd
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

def preprocess_images(raw_images_dir: str, 
                      detections_csv_path: str, 
                      preprocessed_images_dir: str, 
                      preprocessed_csv_path: str, 
                      target_size: int = 224
):
    """
    Crop, resize, and save face images based on detected bounding boxes from a CSV file.

    Each cropped face is resized to a square of `target_size` and saved in the 
    specified output directory. A new CSV file with metadata of preprocessed images 
    is also generated.

    Args:
        raw_images_dir (str): Directory containing the original raw images.
        detections_csv_path (str): Path to the CSV file containing detected faces with columns 
            ['image', 'label', 'x', 'y', 'w', 'h'].
        preprocessed_images_dir (str): Directory where cropped/resized images will be saved.
        preprocessed_csv_path (str): Path to the CSV file which will contain preprocessed 
            images metadata.
        target_size (int, optional): Size of the output square images. Defaults to 224.

    Raises:
        ValueError: If `detections_csv_path` does not point to a CSV file.
        FileNotFoundError: If the CSV file does not exist.
        RuntimeError: If the CSV file is empty.

    Notes:
        - Images are loaded in grayscale.
        - Images with bounding boxes outside the image or invalid crops are skipped with a warning.
        - Output CSV `preprocessed.csv` contains metadata about the new cropped images.
    """
    detected_faces = load_detected_faces(detections_csv_path)

    # Create output directory
    Path(preprocessed_images_dir).mkdir(parents=True, exist_ok=True)

    preprocessed_faces = []
    counter = 0

    # Process each detected face
    for _, row in detected_faces.iterrows():
        metadata = process_single_face(row, raw_images_dir, preprocessed_images_dir, target_size, counter)
        if metadata:
            preprocessed_faces.append(metadata)
            counter += 1

    save_metadata_csv(preprocessed_faces, preprocessed_csv_path)
    logger.info(f"Processed {counter} images. Preprocessed images saved in '{preprocessed_images_dir}'.")


def load_detected_faces(detections_csv_path: str) -> pd.DataFrame:
    """
    Load detected faces from a CSV file.

    Raises:
        ValueError: If the file is not CSV.
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the CSV is empty.
    """
    detections_csv = Path(detections_csv_path)
    if detections_csv.suffix.lower() != ".csv":
        logger.error("The detected faces file has to be in CSV format.")
        raise ValueError("The detected faces file has to be in CSV format.")
    
    if not detections_csv.is_file():
        logger.error(f"Predictions file not found: {detections_csv_path}")
        raise FileNotFoundError(f"Predictions file not found: {detections_csv_path}")
    
    logger.info("Loading detected faces data...")

    detected_faces = pd.read_csv(detections_csv)
    if detected_faces.empty:
        logger.error(f"Detected faces file not found or empty: {detections_csv_path}")
        raise RuntimeError(f"Detected faces file not found or empty: {detections_csv_path}")

    logger.info(f"{len(detected_faces)} detected faces loaded.")

    return detected_faces


def process_single_face(row, raw_images_dir: str, preprocessed_images_dir: str, target_size: int, counter: int):
    """ Crop, resize, and save a single face image. Returns metadata dict or None if failed. """
    file_name = row['image']
    label = row['label']
    x, y, w, h = map(int, (row['x'], row['y'], row['w'], row['h']))

    raw_image_path = Path(raw_images_dir) / file_name
    if not raw_image_path.is_file():
        logger.warning(f"Image not found: {raw_image_path}. Skipping.")
        return None

    img = cv2.imread(raw_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning(f"Error loading image: {raw_image_path}. Skipping.")
        return None

    img_h, img_w = img.shape
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        logger.warning(f"Bounding box outside image: {file_name}. Skipping.")
        return None

    img_cropped = img[y:y + h, x:x + w]
    if img_cropped.size == 0:
        logger.warning(f"Invalid crop: {file_name}. Skipping.")
        return None

    img_resized = cv2.resize(img_cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Save preprocessed image
    img_base = Path(file_name).stem
    img_ext = Path(file_name).suffix
    output_file_name = f"{img_base}_crop_{counter}{img_ext}"
    output_path = Path(preprocessed_images_dir) / output_file_name
    cv2.imwrite(str(output_path), img_resized)

    return {
        "new_image": output_file_name,
        "original_image": file_name,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "label": int(label)
    }


def save_metadata_csv(preprocessed_faces: list[dict], output_csv_name: str):
    """ Save preprocessed faces metadata to CSV. """
    df = pd.DataFrame(preprocessed_faces)
    df.to_csv(output_csv_name, index=False)
    logger.info(f"Labels file saved in '{output_csv_name}'.")
