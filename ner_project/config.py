import torch

class Config:
    # Model Settings
    MODEL_NAME = "dbmdz/bert-base-turkish-cased"
    MAX_LENGTH = 128
    
    # Label Settings
    LABELS = [
        "O", 
        "B-PER", "I-PER", 
        "B-ORG", "I-ORG", 
        "B-LOC", "I-LOC",
        "B-DATE", "I-DATE",
        "B-MONEY", "I-MONEY",
        "B-PERCENT", "I-PERCENT"
    ]
    LABEL2ID = {l: i for i, l in enumerate(LABELS)}
    ID2LABEL = {i: l for l, i in LABEL2ID.items()}
    
    # Training Hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-5
    NUM_EPOCHS = 10
    WEIGHT_DECAY = 0.1
    WARMUP_RATIO = 0.1
    DROPOUT_PROB = 0.2
    EARLY_STOPPING_PATIENCE = 3
    
    # Paths
    DATA_RAW = "data/raw"
    DATA_PROCESSED = "data/processed"
    MODEL_SAVE_PATH = "model"
    LOG_DIR = "logs"
    CHART_DIR = "outputs/charts"
    
    # Features
    CONFIDENCE_THRESHOLD = 0.60
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dataset Generation
    NUM_SYNTHETIC_SAMPLES = 25000
    NEGATIVE_RATIO = 0.25
