import os

# PATHS
FIRE_PATH = os.path.sep.join(["img", "Robbery_Accident_Fire_Database2", "Fire"])
NON_FIRE_PATH = os.path.sep.join(["img", "spatial_envelope_256x256_static_8outdoorcategories"])
MODEL_PATH = os.path.sep.join(["output", "fire_detection.model"])
LRFIND_PLOT_PATH = os.path.sep.join(["output", "lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["output", "training_plot.png"])
OUTPUT_IMAGE_PATH = os.path.sep.join(["output", "examples"])




# CLASSES
CLASSES = ["Non-Fire", "Fire"]

# splits, learning rate, batch size, and number of epochs
TRAIN_SPLIT = 0.75
TEST_SPLIT = 0.25
INIT_LR = 1e-2
BATCH_SIZE = 64
NUM_EPOCHS = 50
SAMPLE_SIZE = 50



