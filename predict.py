# USAGE
# python predict.py
# import the necessary packages
from config import * 
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os

NEVER_SEEN = 0

def prepare_plot(origImage, origMask, predMask):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(origImage)
	ax[1].imshow(origMask)
	ax[2].imshow(predMask)
	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")
	# set the layout of the figure and display it
	figure.tight_layout()
	plt.show()
	

def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0
		# resize the image and make a copy of it for visualization
		image = cv2.resize(image, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
		orig = image.copy()
		# find the filename and generate the path to ground truth
		# mask
		filename = imagePath.split("/")[-1]
		groundTruthPath = MASK_DATASET_PATH + "/" + filename
		if NEVER_SEEN: 
			gtMask = np.zeros((INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, 3), dtype = np.uint8)
		else:
			gtMask = cv2.imread(groundTruthPath, 0)
			gtMask = cv2.resize(gtMask, (INPUT_IMAGE_HEIGHT,
                INPUT_IMAGE_HEIGHT))
		image = np.transpose(image, (2, 0, 1))
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(DEVICE)
		# make the prediction, pass the results through the sigmoid
		# function, and convert the result to a NumPy array
		predMask = model(image).squeeze()
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()
		# filter out the weak predictions and convert them to integers
		predMask2 = predMask*255
		predMask3 = predMask*255
		predMask = (predMask > 1) * 255 #predMask*255 #(predMask > 0.5) * 255
		predMask2 = (predMask2 > 7)
		predMask2 = predMask2.astype(np.uint8)
		predMask = predMask.astype(np.uint8)
		predMask3 = predMask3.astype(np.uint8)
		# prepare a plot for visualization
		prepare_plot(orig, gtMask, predMask3)
		
NEVER_SEEN_PATH = "output/never_seen.txt"
print("[INFO] loading up test image paths...")

imagePaths = open(NEVER_SEEN_PATH).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)
print("[INFO] load up model...")
unet = torch.load(MODEL_PATH).to(DEVICE)
# iterate over the randomly selected test image paths
# for path in imagePaths:
#     # make predictions and visualize the results
#     make_predictions(unet, path)

def plot_histogram(img):
	# tuple to select colors of each channel line
    colors = ("red", "green", "blue")

    # create the histogram plot, with three lines, one for
    # each color
    plt.figure()
    plt.xlim([0, 256])
    for channel_id, color in enumerate(colors):
        histogram, bin_edges = np.histogram(
            img[:, :, channel_id], bins=256, range=(0, 256)
        )
        plt.plot(bin_edges[0:-1], histogram, color=color)

    plt.title("Color Histogram")
    plt.xlabel("Color value")
    plt.ylabel("Pixel count")

def make_my_preds(model, imagePath, maskPath):
	model.eval()
	for filename in os.listdir(imagePath):
		with torch.no_grad():
			path = (imagePath + "/" + filename)
			image = cv2.imread(path)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			image = image.astype("float32") / 255.0
			image = cv2.resize(image, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
			orig = image.copy()
			

			groundTruthMask = (maskPath + "/" + filename)
			gtMask = cv2.imread(groundTruthMask, 0)
			gtMask = cv2.resize(gtMask, (INPUT_IMAGE_HEIGHT,INPUT_IMAGE_HEIGHT))
			cv2.imshow("wind", gtMask)
			cv2.waitKey(0)

			image = np.transpose(image, (2, 0, 1))
			image = np.expand_dims(image, 0)
			image = torch.from_numpy(image).to(DEVICE)
			predMask = model(image).squeeze()
			predMask = torch.sigmoid(predMask)
			predMask = predMask.cpu().numpy()
			predMask = (predMask * 255)
			predMask = predMask.astype(np.uint8)
			cv2.imshow("window", predMask)
			cv2.waitKey(0)
			intersection = np.logical_and(gtMask, predMask)
			union = np.logical_or(gtMask, predMask)
			iou_score = np.sum(intersection) / np.sum(union)
			print("IOU SCORE: ", iou_score) #intersection over union
			prepare_plot(orig, gtMask, predMask)
	
	# model.eval()
	# for filename in os.listdir(imagePath):
	# 	with torch.no_grad():
	# 		path = (imagePath + "/" + filename)
	# 		print(filename)
	# 		image = cv2.imread(path)
	# 		image = image.astype("float32") / 255.0
	# 		image = cv2.resize(image, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
	# 		orig = image.copy()
			

	# 		groundTruthMask = (maskPath + "/" + filename)
	# 		gtMask = cv2.imread(groundTruthMask,0)
	# 		gtMask = cv2.resize(gtMask, (INPUT_IMAGE_HEIGHT,INPUT_IMAGE_HEIGHT))
			
    #         #Showing the binary mask
	# 		cv2.imshow("wind",gtMask)
	# 		cv2.waitKey(0)
			
	# 		image = np.transpose(image, (2, 0, 1))
	# 		image = np.expand_dims(image, 0)
	# 		image = torch.from_numpy(image).to(DEVICE)
	# 		predMask = model(image).squeeze()
			
			
	# 		print("The maximum value of the predictions of the model: ", torch.max(predMask))
	# 		print("The minimum value of the predictions from the model: ", torch.min(predMask))
	# 		predMask = torch.sigmoid(predMask)
	# 		print("The maximum value of the predictions of the model after scaling: " ,torch.max(predMask))
	# 		print("The minimum value of the predictions of the model after scaling: " ,torch.min(predMask))
	# 		predMask = predMask.cpu().numpy()
	# 		predMask = (predMask) * 255
	# 		predMask = predMask.astype(np.uint8)
			
    #         #Showing the predicted mask after scaling
	# 		cv2.imshow("win", predMask)
	# 		cv2.waitKey(0)
			
    #         #Evaluation metrics
	# 		intersection = np.logical_and(gtMask, predMask)
	# 		union = np.logical_or(gtMask, predMask)
	# 		iou_score = np.sum(intersection) / np.sum(union)
	# 		print("IOU SCORE: ", iou_score) #intersection over union
			
    #         #Plot the originals, masks and predictions
	# 		prepare_plot(orig, gtMask, predMask)
            
		
testImgPath = "C:/Users/ingvilrh/OneDrive - NTNU/Masteroppgave23/eyeDetection/testImg"
maskImgPath = "C:/Users/ingvilrh/OneDrive - NTNU/Masteroppgave23/eyeDetection/testMasks"

make_my_preds(unet, testImgPath, maskImgPath)
		
	
