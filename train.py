# USAGE
# python train.py
# import the necessary packages
from dataset import SegmentationDataset
from config import *
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os




# load the image and mask filepaths in a sorted manner
imagePaths = sorted(list(paths.list_images(IMAGE_DATASET_PATH)))
maskPaths = sorted(list(paths.list_images(MASK_DATASET_PATH)))


# partition the data into training and testing splits using 85% of
# the data for training and the remaining 20% for validation
split = train_test_split(imagePaths, maskPaths, test_size=0.2, random_state=42)
# unpack the data split
(trainImages, valImages) = split[:2]

(trainMasks, valMasks) = split[2:]
# write the testing image paths to disk so that we can use then
# when evaluating/testing our model
#print("[INFO] saving testing image paths...")
f = open(TEST_PATHS, "w")
f.write("\n".join(valImages))
f.close()
#DETTE BETYR AT MAN BRUKER FINAL TESTING PÅ SAMME DATA SOM INITIAL TESTING FRA START, SOM ER VELDIG RART

# define transformations
if NET == "UNET":
    transform = transforms.Compose([transforms.ToPILImage(), 
 	                            transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
	                            transforms.ToTensor()])

if NET == "UNET2":
    transform2 = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(num_output_channels=1),
 	                            transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)),
	                            transforms.ToTensor()])


# create the train and validation datasets
trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
	transforms=transform)

valDS = SegmentationDataset(imagePaths=valImages, maskPaths=valMasks,
    transforms=transform)

#print(f"[INFO] found {len(trainDS)} examples in the training set...")
#print(f"[INFO] found {len(valDS)} examples in the validation set...")
# create the training and test data loaders

trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,
	num_workers=os.cpu_count())

valLoader = DataLoader(valDS, shuffle=False,
	batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,
	num_workers=os.cpu_count())



# initialize loss function and optimizer
lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=INIT_LR)
# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // BATCH_SIZE
valSteps = len(valDS) // BATCH_SIZE
# initialize a dictionary to store training history
H = {"train_loss": [], "validation_loss": []}


def main():
    # loop over epochs
    print("[INFO] training the network...", NET)
    startTime = time.time()
    for e in tqdm(range(NUM_EPOCHS)):
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        # loop over the training set
        for (i, (x, y)) in enumerate(trainLoader):
            # set the model in training mode
            unet.train()
            opt.zero_grad()
            # print(x.shape)
            # figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
            # ax[0].imshow(  x[i].permute(1, 2, 0)  )
            # ax[1].imshow(  y[i].permute(1, 2, 0)  )
            # plt.show()
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))
            # perform a forward pass and calculate the training loss
            print(x.shape)
            pred = unet(x)
            loss = lossFunc(pred, y)
            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            
            loss.backward()
            opt.step()
            # add the loss to the total training loss so far
            totalTrainLoss += loss
            #print("loss:", loss)
            #print("total train loss:", totalTrainLoss)
        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            unet.eval()
            # loop over the validation set
            for (x, y) in valLoader:
                # send the input to the device
                (x, y) = (x.to(DEVICE), y.to(DEVICE))
                # make the predictions and calculate the validation loss
                pred = unet(x)
                totalValLoss += lossFunc(pred, y)
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps #her deler jeg med NULL OG DET VIL JEG IKKE
        avgValLoss = totalValLoss / valSteps
        print("train steps", trainSteps)
        print("val steps", valSteps)
        print("avgerage Train loss", avgTrainLoss)
        print("average val loss", avgValLoss)
        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["validation_loss"].append(avgValLoss.cpu().detach().numpy())
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
        print("Train loss: {:.6f}, Validation loss: {:.4f}".format(
            avgTrainLoss, avgValLoss))
    # display the total time needed to perform the training
    endTime = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(
        endTime - startTime))
    



    # plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["validation_loss"], label="validation_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(PLOT_PATH)
    # serialize the model to disk
    torch.save(unet, MODEL_PATH)


if __name__ == "__main__":
    main()