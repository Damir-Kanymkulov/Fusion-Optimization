from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

#-----------------------------------------------------Definition of functions-----------------------------------------------------
#-------------------------------------------Loading dataset part------------------------------------------------------------------
main_folder = "drive/MyDrive/Google_colaboratory/Hackathon_data"
dataset = pd.read_csv(main_folder + "/annotations.csv")
dataset2 = dataset.drop(columns=dataset.columns[5:13])

# Classes
file = open(main_folder + "/classes.txt").readlines()
classes = {}
for line in file:
  key_and_value = line.split("-> ")
  classes[key_and_value[0]] = key_and_value[1].split("\n")[0]

#Data Preprocessing
#Normal
image_dir = main_folder + '/Images/'

left_normal = dataset2[dataset2['Left_Diagnosis'].str.contains('normal')]
selected_left_normal = left_normal[['Age',	'Sex', 'Left_Fundus']].rename(columns={"Left_Fundus": "Filename", "Age": "Label"})
right_normal = dataset2[dataset2['Right_Diagnosis'].str.contains('normal')]
selected_right_normal = right_normal[['Age',	'Sex', 'Right_Fundus']].rename(columns={"Right_Fundus": "Filename", "Age": "Label"})
# normal_image_names = pd.concat([selected_left_normal, selected_right_normal])
# normal_image_names["Filename"] = image_dir + normal_image_names["Filename"]
# normal_image_names = normal_image_names.reset_index(drop=True)
selected_left_normal["Filename"] = image_dir + selected_left_normal["Filename"]
selected_left_normal = selected_left_normal.reset_index(drop=True)
selected_right_normal["Filename"] = image_dir + selected_right_normal["Filename"]
selected_right_normal = selected_right_normal.reset_index(drop=True)

#hypertension
left_hypertension = dataset2[dataset2['Left_Diagnosis'].str.contains('hypertensive retinopathy')]
selected_left_hypertension = left_hypertension[['Age',	'Sex', 'Left_Fundus']].rename(columns={"Left_Fundus": "Filename", "Age": "Label"})
right_hypertension = dataset2[dataset2['Right_Diagnosis'].str.contains('hypertensive retinopathy')]
selected_right_hypertension = right_hypertension[['Age',	'Sex', 'Right_Fundus']].rename(columns={"Right_Fundus": "Filename", "Age": "Label"})
# hypertension_image_names = pd.concat([selected_left_hypertension, selected_right_hypertension])
# hypertension_image_names["Filename"] = image_dir + hypertension_image_names["Filename"]
# hypertension_image_names = hypertension_image_names.reset_index(drop=True)
selected_left_hypertension["Filename"] = image_dir + selected_left_hypertension["Filename"]
selected_left_hypertension = selected_left_hypertension.reset_index(drop=True)
selected_right_hypertension["Filename"] = image_dir + selected_right_hypertension["Filename"]
selected_right_hypertension = selected_right_hypertension.reset_index(drop=True)

#Data collection in two dataframes
df_selected_left_normal = selected_left_normal.copy()
df_selected_left_normal["Diagnosis"] = 0
df_selected_left_normal["Sex"] = df_selected_left_normal["Sex"].replace("Male", 0)
df_selected_left_normal["Sex"] = df_selected_left_normal["Sex"].replace("Female", 1)
df_selected_right_normal = selected_right_normal.copy()
df_selected_right_normal["Diagnosis"] = 0
df_selected_right_normal["Sex"] = df_selected_right_normal["Sex"].replace("Male", 0)
df_selected_right_normal["Sex"] = df_selected_right_normal["Sex"].replace("Female", 1)
df_selected_left_hypertension = selected_left_hypertension.copy()
df_selected_left_hypertension["Diagnosis"] = 1
df_selected_left_hypertension["Sex"] = df_selected_left_hypertension["Sex"].replace("Male", 0)
df_selected_left_hypertension["Sex"] = df_selected_left_hypertension["Sex"].replace("Female", 1)
df_selected_right_hypertension = selected_right_hypertension.copy()
df_selected_right_hypertension["Diagnosis"] = 1
df_selected_right_hypertension["Sex"] = df_selected_right_hypertension["Sex"].replace("Male", 0)
df_selected_right_hypertension["Sex"] = df_selected_right_hypertension["Sex"].replace("Female", 1)

#-------------------------------------------------------------------------Data division---------------------------------------------------------------------------
df_selected_left_normal = df_selected_left_normal.head(2)
df_selected_right_normal = df_selected_right_normal.head(2)
df_selected_left_hypertension = df_selected_left_hypertension.head(2)
df_selected_right_hypertension = df_selected_right_hypertension.head(2)
df_left_combined = pd.concat([df_selected_left_normal, df_selected_left_hypertension]).reset_index(drop=True)
df_right_combined = pd.concat([df_selected_right_normal, df_selected_right_hypertension]).reset_index(drop=True)
#Combined data
df_combined = pd.concat([df_selected_left_normal, df_selected_right_normal, df_selected_right_hypertension, df_selected_left_hypertension]).reset_index(drop=True)

#------------------------------------------------------------Framework only accepts arrays to the input------------------------------------------------------------
# Train = 90%, val = 10%
X_left_train, X_left_test, y_left_train, y_left_test = train_test_split(df_left_combined[["Diagnosis", "Sex", "Filename"]], df_left_combined["Label"], test_size=0.1, random_state=42)
X_right_train, X_right_test, y_right_train, y_right_test = train_test_split(df_right_combined[["Diagnosis", "Sex", "Filename"]], df_right_combined["Label"], test_size=0.1, random_state=42)
# train_left_AttrX, test_left_AttrX, train_left_ImagesX, test_left_ImagesX = np.array(X_left_train[["Diagnosis", "Sex"]]), np.array(X_left_test[["Diagnosis", "Sex"]]), np.array(X_left_train["Filename"]), np.array(X_left_test["Filename"])
train_left_NumX, test_left_NumX, train_left_ImagesX, test_left_ImagesX = np.array(X_left_train["Sex"]), np.array(X_left_test["Sex"]), np.array(X_left_train["Filename"]), np.array(X_left_test["Filename"])
train_left_TextX, test_left_TextX = np.array(X_left_train["Diagnosis"]), np.array(X_left_test["Diagnosis"])
# train_right_AttrX, test_right_AttrX, train_right_ImagesX, test_right_ImagesX = np.array(X_right_train[["Diagnosis", "Sex"]]), np.array(X_right_test[["Diagnosis", "Sex"]]), np.array(X_right_train["Filename"]), np.array(X_right_test["Filename"])
train_right_NumX, test_right_NumX, train_right_ImagesX, test_right_ImagesX = np.array(X_right_train["Sex"]), np.array(X_right_test["Sex"]), np.array(X_right_train["Filename"]), np.array(X_right_test["Filename"])
train_right_TextX, test_right_TextX = np.array(X_right_train["Diagnosis"]), np.array(X_right_test["Diagnosis"])
new_trainY = np.array(y_left_train)
new_testY = np.array(y_left_test)

#Combined dataset
X_train, X_test, y_train, y_test = train_test_split(df_combined[["Diagnosis", "Sex", "Filename"]], df_combined["Label"], test_size=0.1, random_state=42)
train_AttrX, test_AttrX, train_ImagesX, test_ImagesX = np.array(X_train[["Diagnosis", "Sex"]]), np.array(X_test[["Diagnosis", "Sex"]]), np.array(X_train["Filename"]), np.array(X_test["Filename"])
combined_trainY = np.array(y_train)
combined_testY = np.array(y_test)

#---------------------------------------------------------------Datasets and Dataloaders--------------------------------------------------------------------------
class ImageData(Dataset):
    def __init__(self, path_array, labels, transform):
        self.path_array = path_array
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.path_array.shape[0]

    def __getitem__(self, index):
        PIL_image = Image.open(self.path_array[index])
        transformed_image = self.transform(PIL_image)
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

class CombinedData(Dataset):
    def __init__(self, path_array, attributes, labels, transform):
        self.path_array = path_array
        self.attributes = attributes
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.path_array.shape[0]

    def __getitem__(self, index):
        PIL_image = Image.open(self.path_array[index])
        transformed_image = self.transform(PIL_image)
        transformed_image = transformed_image.view(-1)
        attribute = torch.from_numpy(self.attributes[index])
        transformed_attribute = attribute.view(-1)
        combined_data = torch.cat([transformed_image, transformed_attribute], dim=0)
        label = self.labels[index]
        return combined_data, label

#-----------------------------------------------------------------------Main part---------------------------------------------------------------------------------
batch_size = 2
n_loaders = os.cpu_count()
mean = torch.tensor([0.5], dtype=torch.float32)
std = torch.tensor([1], dtype=torch.float32)

train_transf = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(list(mean), list(std))
])
val_transf = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(list(mean), list(std))
])

#Early fusion train data
train_combined_data = CombinedData(train_ImagesX, train_AttrX, combined_trainY, train_transf)
train_combined_loader = torch.utils.data.DataLoader(train_combined_data, batch_size=batch_size, shuffle=True, num_workers=n_loaders)

#Early fusion train data
val_combined_data = CombinedData(test_ImagesX, test_AttrX, combined_testY, val_transf)
val_combined_loader = torch.utils.data.DataLoader(val_combined_data, batch_size=batch_size, shuffle=False, num_workers=n_loaders)

#Intermediate and late fusion train data
# train_left_attr_data = AttrData(train_left_AttrX, new_trainY)
# train_right_attr_data = AttrData(train_right_AttrX, new_trainY)
train_left_num_data = AttrData(train_left_NumX, new_trainY)
train_right_num_data = AttrData(train_right_NumX, new_trainY)
train_left_text_data = AttrData(train_left_TextX, new_trainY)
train_right_text_data = AttrData(train_right_TextX, new_trainY)
train_left_image_data = ImageData(train_left_ImagesX, new_trainY, train_transf)
train_right_image_data = ImageData(train_right_ImagesX, new_trainY, train_transf)
# train_left_attr_loader = torch.utils.data.DataLoader(train_left_attr_data, batch_size=batch_size, shuffle=True, num_workers=n_loaders)
# train_right_attr_loader = torch.utils.data.DataLoader(train_right_attr_data, batch_size=batch_size, shuffle=True, num_workers=n_loaders)
train_left_num_loader = torch.utils.data.DataLoader(train_left_num_data, batch_size=batch_size, shuffle=True, num_workers=n_loaders)
train_right_num_loader = torch.utils.data.DataLoader(train_right_num_data, batch_size=batch_size, shuffle=True, num_workers=n_loaders)
train_left_text_loader = torch.utils.data.DataLoader(train_left_text_data, batch_size=batch_size, shuffle=True, num_workers=n_loaders)
train_right_text_loader = torch.utils.data.DataLoader(train_right_text_data, batch_size=batch_size, shuffle=True, num_workers=n_loaders)
train_left_image_loader = torch.utils.data.DataLoader(train_left_image_data, batch_size=batch_size, shuffle=True, num_workers=n_loaders)
train_right_image_loader = torch.utils.data.DataLoader(train_right_image_data, batch_size=batch_size, shuffle=True, num_workers=n_loaders)

#Intermediate and late fusion train data
# val_left_attr_data = AttrData(test_left_AttrX, new_testY)
# val_right_attr_data = AttrData(test_right_AttrX, new_testY)
val_left_num_data = AttrData(test_left_NumX, new_testY)
val_right_num_data = AttrData(test_right_NumX, new_testY)
val_left_text_data = AttrData(test_left_TextX, new_testY)
val_right_text_data = AttrData(test_right_TextX, new_testY)
val_left_image_data = ImageData(test_left_ImagesX, new_testY, val_transf)
val_right_image_data = ImageData(test_right_ImagesX, new_testY, val_transf)
# val_left_attr_loader = torch.utils.data.DataLoader(val_left_attr_data, batch_size=batch_size, shuffle=False, num_workers=n_loaders)
# val_right_attr_loader = torch.utils.data.DataLoader(val_right_attr_data, batch_size=batch_size, shuffle=False, num_workers=n_loaders)
val_left_num_loader = torch.utils.data.DataLoader(val_left_num_data, batch_size=batch_size, shuffle=False, num_workers=n_loaders)
val_right_num_loader = torch.utils.data.DataLoader(val_right_num_data, batch_size=batch_size, shuffle=False, num_workers=n_loaders)
val_left_text_loader = torch.utils.data.DataLoader(val_left_text_data, batch_size=batch_size, shuffle=False, num_workers=n_loaders)
val_right_text_loader = torch.utils.data.DataLoader(val_right_text_data, batch_size=batch_size, shuffle=False, num_workers=n_loaders)
val_left_image_loader = torch.utils.data.DataLoader(val_left_image_data, batch_size=batch_size, shuffle=False, num_workers=n_loaders)
val_right_image_loader = torch.utils.data.DataLoader(val_right_image_data, batch_size=batch_size, shuffle=False, num_workers=n_loaders)

loaders_dict = {"numerical": [], "text": [], "image": []}
# loaders_dict["attribute"].append((train_left_attr_loader, val_left_attr_loader))
# loaders_dict["attribute"].append((train_right_attr_loader, val_right_attr_loader))
loaders_dict["numerical"].append((train_left_num_loader, val_left_num_loader))
loaders_dict["numerical"].append((train_right_num_loader, val_right_num_loader))
loaders_dict["text"].append((train_left_text_loader, val_left_text_loader))
loaders_dict["text"].append((train_right_text_loader, val_right_text_loader))
loaders_dict["image"].append((train_left_image_loader, val_left_image_loader))
loaders_dict["image"].append((train_right_image_loader, val_right_image_loader))

# attr_dim = train_left_attr_data[0][0].shape[0]
num_dim = len(train_left_num_data[0]) - 1
text_dim = len(train_left_text_data[0]) - 1
img_dim = train_left_image_data[0][0].shape[0] * train_left_image_data[0][0].shape[1] * train_left_image_data[0][0].shape[2]
dimension_dict = {}
# dimension_dict["attribute"] = attr_dim
dimension_dict["numerical"] = num_dim
dimension_dict["text"] = text_dim
dimension_dict["image"] = img_dim