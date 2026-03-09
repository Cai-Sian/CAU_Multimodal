#!/bin/python3

#Featuremap extraction

import logging
import sys
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.special import softmax
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score, recall_score,accuracy_score,f1_score, ConfusionMatrixDisplay
import seaborn as sns
import monai
from monai.config import print_config
from monai.data import decollate_batch, ImageDataset,DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121,resnet
import torch.nn as nn
import pandas as pd
from monai.utils import set_determinism

from monai.transforms import (
    EnsureChannelFirst,
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    EnsureType,
    Resized,
    Resize,
    RandRotate90,
)
#################

img_dir = '../image_inputs/preprocess' # where the image saved
img_type = 'preprocessed'

image_codebook = '../image_inputs/CNN_sample_split_paper.txt'

model_dir = '../Results/Medical_images'
model = 'ResNet50'
model_name = 'model_ResNet50_5000epochs.pth'

model_pth = f'{model_dir}/{model_name}'

####################
set_determinism(seed=0)

print_config()
device = torch.device("cuda:0")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print_config()

allimg_label = pd.read_csv(image_codebook,sep='\t')
print(allimg_label)
allimg_label[img_type] = f'{img_dir}/{allimg_label['img_name'].astype(str)}'

image_Data_x=allimg_label[img_dir].tolist()
image_Data_y=allimg_label['caco'].tolist()

val_transforms = Compose([EnsureChannelFirst(),ScaleIntensity(),Resize([600,216]), EnsureType()])

allimg_ds = ImageDataset(image_files=image_Data_x, labels=image_Data_y, transform = val_transforms)
allimg_loader = DataLoader(allimg_ds, batch_size=1,num_workers=4,pin_memory=torch.cuda.is_available())

if model == 'DenseNet121':
    model = DenseNet121(spatial_dims=2, in_channels=3,out_channels=2).to(device)
    print(model_pth)
    model.load_state_dict(torch.load(os.path.join(model_pth)))
    del model.class_layers.out

elif model == 'ResNet50':
    model = resnet.ResNet(block = 'bottleneck',layers = [3,4,6,3], block_inplanes = resnet.get_inplanes(),spatial_dims=2, n_input_channels=3,num_classes=2).to(device)
    print(model_pth)
    model.load_state_dict(torch.load(os.path.join(model_pth)))
    model.fc = nn.Identity() 
    
model.eval()
with torch.no_grad():
    featuremap_allimg = torch.tensor([], dtype=torch.float32, device=device)
    for allimg_data in allimg_loader:
        allimg_images, allimg_labels = (
            allimg_data[0].to(device),
            allimg_data[1].to(device),
        )
        feature = model(allimg_images)
        featuremap_allimg = torch.cat((featuremap_allimg, feature), 0)
        
allimg_featuremap=pd.DataFrame(featuremap_allimg.to("cpu").numpy())
allimg_featuremap['ID'] = image_Data_x

model_name_1 = model_name.strip('.pth')
allimg_featuremap.to_csv(f'../multimodal_inputs/{model_name_1}_extract_feature.txt',sep='\t')
