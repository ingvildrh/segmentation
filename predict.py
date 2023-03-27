# USAGE
# python predict.py
# import the necessary packages
from config import * 
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
from train import *
from torchvision import transforms
from PIL import Image

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
	
		
NEVER_SEEN_PATH = "output/never_seen.txt"
print("[INFO] loading up test image paths...")

imagePaths = open(NEVER_SEEN_PATH).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)
print("[INFO] load up model...")

unet = torch.load(MODEL_PATH).to(DEVICE)


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
			#ret, tresh = cv2.threshold(predMask, 1,255,0) creating a threshold value so I can see a binary plot of the mask
			cv2.imshow("window", predMask)
			cv2.waitKey(0)
			intersection = np.logical_and(gtMask, predMask)
			union = np.logical_or(gtMask, predMask)
			iou_score = np.sum(intersection) / np.sum(union)
			print("IOU SCORE: ", iou_score) #intersection over union
			prepare_plot(orig, gtMask, predMask)
	

def predict_unet2(testImgPath, maskImgPath, model):
	testImgPath = sorted(list(paths.list_images(testImgPath)))
	maskImgPath = sorted(list(paths.list_images(maskImgPath)))
	testDS = SegmentationDataset(testImgPath, maskImgPath, transform)
	testLoader = DataLoader(testDS, shuffle=True,
	batch_size=2, pin_memory=PIN_MEMORY)
	model.eval()
	length = testDS.__len__()
	for i in range(length):
		img, msk = testDS.__getitem__(i)
		img, mask = img.to(DEVICE), msk.to(DEVICE)
		img = img.unsqueeze(0)
		pred = model(img)
		pred = torch.sigmoid(pred)
		
		pred = pred*255
		pred = pred.detach().cpu().numpy()
		pred = pred.reshape(256,256)

		prepare_plot((img.reshape(256,256)).cpu(), (mask.reshape(256,256).cpu()), pred)
	


testImgPath = "C:/Users/ingvilrh/OneDrive - NTNU/Masteroppgave23/eyeDetection/testImg"
maskImgPath = "C:/Users/ingvilrh/OneDrive - NTNU/Masteroppgave23/eyeDetection/testMasks"

def main():
    if NET == "UNET2":
        predict_unet2(testImgPath, maskImgPath, unet)
    if NET == "UNET":
        make_my_preds(unet, testImgPath, maskImgPath)
#make_my_preds(unet, testImgPath, maskImgPath)
		
	
if __name__ == "__main__":
    main()