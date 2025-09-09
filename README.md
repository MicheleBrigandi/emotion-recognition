# Face Emotion Recognition

This project performs **face emotion recognition** using a pre-trained deep learning model
(built with **TensorFlow/Keras**) and image processing with **OpenCV**.  

The pipeline covers:
- Preprocessing of face images from raw detections  
- Running predictions with a pre-trained model  
- Saving results in CSV format  
- Visualisation with bounding boxes and predicted labels on faces  

⚠️ **Note**: Model training is not part of this pipeline. A Jupyter Notebook for training
is included in the repository, but the code here is designed only for **inference**.

---

## 📂 Project Structure

.
├── .venv/                      # Python virtual environment  
├── config/  
│   ├── config.py               # Global configuration class  
│   └── config.yaml             # Configuration file with paths, image size, etc.  
├── data/  
│   ├── test/  
│   │   ├── raw/images/         # Raw images for testing  
│   │   ├── raw/detections.csv  # Detected bounding boxes for raw images  
│   │   ├── preprocessed/images/ # Preprocessed face images  
│   │   └── preprocessed/processed.csv # Metadata for preprocessed images  
├── logs/  
│   └── pipeline.log            # Log file  
├── models/  
│   ├── custom_objects/         # Custom layers (SEBlock, CAMBlock, etc.)  
│   └── weights/                # Pre-trained model weights  
├── preprocessing/  
│   └── image_preprocessing.py  # Functions to crop, resize, and save face images  
├── results/                     # Outputs (ignored by Git)  
│   ├── annotated/               # Annotated images with bounding boxes  
│   ├── metrics/                 # Saved evaluation metrics and confusion matrices  
│   └── predictions.csv          # Predictions CSV  
├── utils/  
│   ├── visualization.py         # Drawing bounding boxes and labels  
│   ├── metrics_evaluation.py    # Accuracy, classification report, confusion matrix  
│   └── logger.py                # Logger setup  
├── notebooks/                   # Jupyter notebooks for training models (reference only)  
├── main.py                      # Pipeline entry point  
├── .gitignore  
├── requirements.txt  
└── README.md

---

## 🚀 Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

### 2. Install dependencies

```pip install -r requirements.txt```

## 🔮 Running Predictions

Run the full pipeline (preprocessing + predictions + visualization):

```python main.py```

This will:
- Check if preprocessing is needed and run it if necessary
- Load the pre-trained model from models/weights/
- Run predictions on the preprocessed images
- Save results in results/predictions.csv

## 🖼️ Visualisation
Annotated images with bounding boxes and predicted labels are saved in:
```results/annotated/```

✅ Green boxes indicate correct predictions
❌ Red boxes indicate incorrect predictions

## ⚙️ Configuration
All paths, image sizes, logger settings, and model options are configurable in:
```config/config.yaml```

The Config class (```config/config.py```) loads these settings and ensures a single global instance.