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
from PIL import Image

#-------------------------------------------Dataset and Dataloader part-------------------------------------------------------------
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

#--------------------------------------------------------Main part--------------------------------------------------------------------
from sklearn import preprocessing

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

train_df = pd.read_csv('/content/drive/MyDrive/Google_colaboratory/Lab rotation prof. Heider/shopee-product-matching/train.csv')
test_df = pd.read_csv('/content/drive/MyDrive/Google_colaboratory/Lab rotation prof. Heider/shopee-product-matching/test.csv')
train_image_path='/content/drive/MyDrive/Google_colaboratory/Lab rotation prof. Heider/shopee-product-matching/train_images/'
test_image_path='/content/drive/MyDrive/Google_colaboratory/Lab rotation prof. Heider/shopee-product-matching/test_images/'

train_df['image_path'] = train_df['image'].apply(lambda x: train_image_path + str(x))
labels_list = train_df['label_group'].value_counts().index[:100].tolist() ##filtering only 100 labels
train_df = train_df[train_df['label_group'].isin(labels_list)].reset_index(drop=True)
le = preprocessing.LabelEncoder()
train_df['label'] = le.fit_transform(train_df['label_group'])

train_shopee_data, val_shopee_data = train_test_split(train_df, test_size=0.1, random_state=42)

#----------------------------------------------Image shopee dataloaders----------------------------------------------
train_shopee_image_dataset = ImageData(train_shopee_data['image_path'].values, train_shopee_data['label'].values, train_transf)
val_shopee_image_dataset = ImageData(val_shopee_data['image_path'].values, val_shopee_data['label'].values, val_transf)
train_shopee_image_loader = torch.utils.data.DataLoader(train_shopee_image_dataset, batch_size=batch_size, shuffle=True, num_workers=n_loaders)
val_shopee_image_loader = torch.utils.data.DataLoader(val_shopee_image_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders)

#----------------------------------------------Text shopee dataloaders----------------------------------------------
train_shopee_text_data = train_shopee_data['title'].dropna().values
val_shopee_text_data = val_shopee_data['title'].dropna().values
y_train = train_shopee_data['label'].values
y_test = val_shopee_data['label'].values

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(train_shopee_text_data)
train_shopee_text_data = vectorizer.transform(train_shopee_text_data).toarray()
val_shopee_text_data  = vectorizer.transform(val_shopee_text_data).toarray()

train_shopee_text_dataset = AttrData(train_shopee_text_data, y_train)
val_shopee_text_dataset = AttrData(val_shopee_text_data, y_test)
train_shopee_text_loader = torch.utils.data.DataLoader(train_shopee_text_dataset, batch_size=batch_size, shuffle=True, num_workers=n_loaders)
val_shopee_text_loader = torch.utils.data.DataLoader(val_shopee_text_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders)

#----------------------------------------------Dataloaders' dictionary----------------------------------------------
loaders_dict = {"text": [], "image": []}
loaders_dict["text"].append((train_shopee_text_loader, val_shopee_text_loader))
loaders_dict["image"].append((train_shopee_image_loader, val_shopee_image_loader))

dimension_dict = {}
text_dim = train_shopee_text_dataset[0][0].shape[0]
dimension_dict["text"] = text_dim
image_dim = train_shopee_image_dataset[0][0].shape[0] * train_shopee_image_dataset[0][0].shape[1] * train_shopee_image_dataset[0][0].shape[2]
dimension_dict["image"] = image_dim

#----------------------------------------------Early fusion combined data----------------------------------------------
train_shopee_combined_dataset = CombinedData(train_shopee_data['image_path'].values, train_shopee_text_data, train_shopee_data['label'].values, train_transf)
train_shopee_combined_loader = torch.utils.data.DataLoader(train_shopee_combined_dataset, batch_size=batch_size, shuffle=True, num_workers=n_loaders)
val_shopee_combined_dataset = CombinedData(val_shopee_data['image_path'].values, val_shopee_text_data, val_shopee_data['label'].values, val_transf)
val_shopee_combined_loader = torch.utils.data.DataLoader(val_shopee_combined_dataset, batch_size=batch_size, shuffle=False, num_workers=n_loaders)