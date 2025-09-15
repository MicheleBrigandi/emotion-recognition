# Michele Brigandì - 2156373

import yaml

class Config:
    """
    Singleton class that loads and stores pipeline configuration from a YAML file.

    This class ensures only one instance is created and reused throughout the program.
    It reads the configuration once (on the first instantiation) and provides access 
    to various paths, preprocessing options, emotion labels, and logging parameters.

    Attributes:
        model_path (str): Path to the trained model file.
        raw_images_folder (str): Directory containing raw input images.
        detections_csv (str): Path to the CSV file for detections.
        preprocessed_images_folder (str): Directory for preprocessed images.
        preprocessed_csv (str): Path to the CSV file for preprocessed data.
        result_images_folder (str): Directory where annotated images are stored.
        result_metrics_folder (str): Directory where model metrics are stored.
        predictions_csv (str): Path to the CSV file for predictions.
        normalize_input (bool): Whether to normalize input images.
        image_size (tuple): Size (width, height) of input images.
        emotion_labels (dict[int, str]): Mapping from class index to emotion label.
        logger_* (various): Logging configuration options.
    """
    _instance = None

    def __new__(cls):
        """ Ensure only one instance of Config exists (singleton pattern). """
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, path: str = "config/config.yaml"):
        """
        Initialise the configuration by reading from the given YAML file.

        Args:
            path (str): Path to the YAML configuration file. Defaults to 'config/config.yaml'.
        """
        if self._initialized:
            return

        with open(path, "r") as file:
            cfg = yaml.safe_load(file)

        # Paths
        self.model_path: str = cfg["model_path"]
        self.raw_images_folder: str = cfg["raw_images_folder"]
        self.detections_csv: str = cfg["detections_csv"]
        self.preprocessed_images_folder: str = cfg["preprocessed_images_folder"]
        self.preprocessed_csv: str = cfg["preprocessed_csv"]
        self.result_images_folder: str = cfg["result_images_folder"]
        self.result_metrics_folder: str = cfg["result_metrics_folder"]
        self.predictions_csv: str = cfg["predictions_csv"]

        # Preprocessing
        self.normalize_input: bool = cfg.get("normalize_input", True)
        self.image_size: tuple = tuple(cfg["image_size"])

        # Emotions mapping
        self.emotion_labels: dict[int, str] = cfg["emotion_labels"]

        # Logger config
        logger_cfg = cfg.get("logger", {})
        self.logger_level: str = logger_cfg.get("level", "INFO")
        self.logger_file_path: str = logger_cfg.get("file_path", "logs/pipeline.log")
        self.logger_max_file_size_mb: int = logger_cfg.get("max_file_size_mb", 5)
        self.logger_backup_count: int = logger_cfg.get("backup_count", 3)
        self.logger_enable_console: bool = logger_cfg.get("enable_console", True)
        self.logger_enable_file: bool = logger_cfg.get("enable_file", True)

        self._initialized = True

    def __str__(self):
        """ Return a string representation of the configuration (short summary). """
        return f"<Config model={self.model_path}, images={self.raw_images_folder}>"
