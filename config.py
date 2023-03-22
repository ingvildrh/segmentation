'''
This is from https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
'''

# import the necessary packages
import torch
import os
# base path of the dataset
#DATASET_PATH = os.path.join("dataset", "train")
# define the path to the images and masks dataset
IMAGE_DATASET_PATH = "C:/Users/ingvilrh/OneDrive - NTNU/Masteroppgave23/fishForUNETtut/tail_UNET1/train_images/"
MASK_DATASET_PATH = "C:/Users/ingvilrh/OneDrive - NTNU/Masteroppgave23/fishForUNETtut/tail_UNET1/train_masks/"
# define the test split
TEST_SPLIT = 0.15
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False


# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 3
NUM_CLASSES = 1
NUM_LEVELS = 3
# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 2
# define the input image dimensions
INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256
# define threshold to filter weak predictions
THRESHOLD = 0.5
# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_tails.pth")
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS =  "output/test_paths.txt"

