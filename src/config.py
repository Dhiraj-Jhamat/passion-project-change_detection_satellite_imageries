import os

# DATASET PATHS

# Set via environment variable or change the fallback path below
DATASET_ROOT = os.environ.get(
    "OSCD_DATASET_ROOT",
    r"D:\UCDNET\onera-satellite-change-detection-dataset"   
)

IMAGES_ROOT = os.path.join(DATASET_ROOT, "images")
LABELS_ROOT = os.path.join(DATASET_ROOT, "train_labels")

# ALL 24 CITIES 
ALL_CITIES = [
    "aguasclaras", "bercy", "bordeaux",
    "nantes", "paris", "rennes", "saclay_e",
    "abudhabi", "cupertino", "pisa", "beihai",
    "hongkong", "beirut", "mumbai", "brasilia",
    "montpellier", "norcia", "rio", "saclay_w",
    "valencia", "dubai", "lasvegas", "milano",
    "chongqing",
]

# PATCH SPLIT COUNTS
N_TRAIN_PATCHES = 200
N_VAL_PATCHES   = 57
N_TEST_PATCHES  = 30

#  TRAINING HYPER-PARAMETERS 
INPUT_SHAPE   = (512, 512, 13)   
NUM_CLASSES   = 2                
BATCH_SIZE    = 1                
EPOCHS        = 60               
LEARNING_RATE = 0.0001           

# LOSS FUNCTION WEIGHTS
CLASS_WEIGHTS = (0.1, 0.9)

# OUTPUT PATHS
OUTPUT_DIR      = "outputs"
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model.keras")
CURVES_PATH     = os.path.join(OUTPUT_DIR, "training_curves.png")
METRICS_PATH    = os.path.join(OUTPUT_DIR, "metrics.csv")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")

os.makedirs(OUTPUT_DIR,      exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)