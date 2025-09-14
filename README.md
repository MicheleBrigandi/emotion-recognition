# Face Emotion Recognition

This repository provides a **face emotion recognition** pipeline built with
**TensorFlow/Keras** (for the CNN model) and **OpenCV** (for image processing).

The code is designed for **inference only**:
- Detects and preprocesses face images
- Runs predictions with a trained CNN
- Saves results in CSV format
- Generates annotated images with bounding boxes and predicted labels

⚠️ *Note*: Model training is not part of the pipeline. A reference Jupyter Notebook for training is included in the ```notebooks/``` folder.

---

## 📂 Project Structure

```bash
.
├── .venv/                       # Python virtual environment  
├── config/  
│   ├── config.py                # Global configuration class  
│   └── config.yaml              # Configuration file with paths, image size, etc.  
├── data/  
│   └── test/                    # Test dataset (raw + preprocessed)  
│       ├── raw/                 # Original test images + detections  
│       └── preprocessed/        # Cropped and resized face images 
├── logs/  
│   └── pipeline.log             # Log file  
├── models/  
│   ├── custom_objects/          # Custom layers (SEBlock, CAMBlock, etc.)  
│   └── weights/                 # Pre-trained model weights  
├── preprocessing/  
│   └── image_preprocessing.py   # Functions to crop, resize, and save face images  
├── results/                     # Outputs 
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
```

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

The script will:
- Run preprocessing if not already done
- Load the CNN model (from ```models/weights/``` or download from HuggingFace if missing)
- Perform predictions on the preprocessed images
- Save results to ```results/predictions.csv```

## 🖼️ Visualisation
Annotated test images with bounding boxes and predicted labels are saved in:
```results/annotated/```

✅ Green boxes indicate correct predictions
❌ Red boxes indicate incorrect predictions

## ⚙️ Configuration
All settings (paths, image size, logger, model options) are defined in:
```config/config.yaml```

The Config class (```config/config.py```) loads these settings and ensures a single global instance.