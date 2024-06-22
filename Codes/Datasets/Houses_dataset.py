from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

#-----------------------------------------------------Definition of functions-----------------------------------------------------
#-------------------------------------------Loading dataset part------------------------------------------------------------------
def load_house_attributes(inputPath):
	# initialize the list of column names in the CSV file and then
	# load it using Pandas
	cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
	df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)
	# determine (1) the unique zip codes and (2) the number of data
	# points with each zip code
	zipcodes = df["zipcode"].value_counts().keys().tolist()
	counts = df["zipcode"].value_counts().tolist()
	# loop over each of the unique zip codes and their corresponding
	# count
	for (zipcode, count) in zip(zipcodes, counts):
		# the zip code counts for our housing dataset is *extremely*
		# unbalanced (some only having 1 or 2 houses per zip code)
		# so let's sanitize our data by removing any houses with less
		# than 25 houses per zip code
		if count < 25:
			idxs = df[df["zipcode"] == zipcode].index
			df.drop(idxs, inplace=True)
	# return the data frame
	return df.head(4)

# This loads the data and concatenates the four images into one
def load_house_images(df, inputPath):
	# initialize our images array (i.e., the house images themselves)
	images = []
	# loop over the indexes of the houses
	for i in df.index.values:
		# find the four images for the house and sort the file paths,
		# ensuring the four are always in the *same order*
		basePath = os.path.sep.join([inputPath, "{}_*".format(i + 1)])
		housePaths = sorted(list(glob.glob(basePath)))
		# initialize our list of input images along with the output image
		# after *combining* the four input images
		inputImages = []
		outputImage = np.zeros((64, 64, 3), dtype="uint8")
		# loop over the input house paths
		for housePath in housePaths:
			# load the input image, resize it to be 32 32, and then
			# update the list of input images
			image = cv2.imread(housePath)
			image = cv2.resize(image, (32, 32))
			inputImages.append(image)
		# tile the four input images in the output image such the first
		# image goes in the top-right corner, the second image in the
		# top-left corner, the third image in the bottom-right corner,
		# and the final image in the bottom-left corner
		outputImage[0:32, 0:32] = inputImages[0]
		outputImage[0:32, 32:64] = inputImages[1]
		outputImage[32:64, 32:64] = inputImages[2]
		outputImage[32:64, 0:32] = inputImages[3]
		# add the tiled image to our set of images the network will be
		# trained on
		images.append(outputImage)
	# return our set of images
	return np.array(images)

#-------------------------------------------Preprocessing part------------------------------------------------------------------
# Process that data and get reshaped data
def process_house_attributes(df, train, test):
    # initialize the column names of the continuous data
    continuous = ["bedrooms", "bathrooms", "area"]
    # performin min-max scaling each continuous feature column to
    # the range [0, 1]
    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(train[continuous])
    testContinuous = cs.transform(test[continuous])
    # one-hot encode the zip code categorical data (by definition of
    # one-hot encoding, all output features are now in the range [0, 1])
    zipBinarizer = LabelBinarizer().fit(df["zipcode"])
    trainCategorical = zipBinarizer.transform(train["zipcode"])
    testCategorical = zipBinarizer.transform(test["zipcode"])
    # construct our training and testing data points by concatenating
    # the categorical features with the continuous features
    # trainX = np.hstack([trainCategorical, trainContinuous])
    trainX = trainContinuous
    # testX = np.hstack([testCategorical, testContinuous])
    testX = testContinuous
    # return the concatenated training and testing data
    return (trainX, testX)

def preprocessing_part(df_house_att, df_house_img):
    # norm the data to be between 0 and 1 (We can do it via transform toTensor())
    # df_house_img = df_house_img / 255.0
    # training set - 75%; testing set - 25%
    trainAttrX, testAttrX, trainImagesX, testImagesX = train_test_split(df_house_att, df_house_img, test_size=0.25, random_state=42)
    # scale our house prices to the range [0, 1]
    maxPrice = trainAttrX["price"].max()
    trainY = trainAttrX["price"] / maxPrice
    testY = testAttrX["price"] / maxPrice
    # min-max scaling on continuous features, one-hot encoding on categorical features and a final concatenation
    trainAttrX, testAttrX = process_house_attributes(df_house_att, trainAttrX, testAttrX)

    return trainAttrX, testAttrX, trainImagesX, testImagesX, trainY, testY

#-------------------------------------------Dataset and Dataloader part-------------------------------------------------------------
class ImageData(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    transformations = transforms.Compose([
        transforms.ToTensor(),
    ])

    def __getitem__(self, index):
        image = self.images[index]
        transformed_image = self.transformations(image)
        label = self.labels[index]
        return transformed_image, label

class AttrData(Dataset):
    def __init__(self, attributes, labels):
        self.attributes = attributes
        self.labels = labels

    def __len__(self):
        return self.attributes.shape[0]

    def __getitem__(self, index):
        attribute = self.attributes[index]
        label = self.labels[index]
        return attribute, label

class CombinedDataset(Dataset):
    def __init__(self, images, attributes, labels):
        self.images = images
        self.attributes = attributes
        self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    transformations = transforms.Compose([
        transforms.ToTensor(),
    ])

    def __getitem__(self, index):
        image = self.images[index]
        transformed_image = self.transformations(image)
        transformed_image = transformed_image.view(-1)
        attribute = torch.from_numpy(self.attributes[index])
        transformed_attribute = attribute.view(-1)
        img_attr = torch.cat([transformed_image, transformed_attribute], dim=0)
        label = self.labels[index]
        return img_attr, label
    
#--------------------------------------------------------Main part--------------------------------------------------------------------
inputPath_house_data = r"Houses-dataset\Houses Dataset\HousesInfo.txt"
df_house_att = load_house_attributes(inputPath_house_data)
df_house_img = load_house_images(df_house_att, "Houses-dataset/Houses Dataset")

trainAttrX, testAttrX, trainImagesX, testImagesX, trainY, testY = preprocessing_part(df_house_att, df_house_img)
trainY = np.array(trainY)
testY = np.array(testY)

n_loaders = os.cpu_count()
# Create early fusion Datasets and DataLoaders
train_dataset = CombinedDataset(trainImagesX, trainAttrX, trainY)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=n_loaders)
val_dataset = CombinedDataset(testImagesX, testAttrX, testY)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=n_loaders)

# Create intermediate/late fusion Datasets and DataLoaders
train_image_data = ImageData(trainImagesX, trainY)
train_attr_data = AttrData(trainAttrX, trainY)
train_image_loader = torch.utils.data.DataLoader(train_image_data, batch_size=2, shuffle=True, num_workers=n_loaders)
train_attr_loader = torch.utils.data.DataLoader(train_attr_data, batch_size=2, shuffle=True, num_workers=n_loaders)

val_image_data = ImageData(testImagesX, testY)
val_attr_data = AttrData(testAttrX, testY)
val_image_loader = torch.utils.data.DataLoader(val_image_data, batch_size=2, shuffle=False, num_workers=n_loaders)
val_attr_loader = torch.utils.data.DataLoader(val_attr_data, batch_size=2, shuffle=False, num_workers=n_loaders)

img_dim = trainImagesX.shape[1] * trainImagesX.shape[2] * trainImagesX.shape[3]
attr_dim = trainAttrX.shape[1]