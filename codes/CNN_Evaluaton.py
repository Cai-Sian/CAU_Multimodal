#!/bin/python3

## Evaluation

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

print_config()

####

img_dir = '../image_inputs/preprocess'
img_type = 'preprocessed'

image_codebook = '../image_inputs/CNN_sample_split_paper.txt'

model_dir = '../Results/Medical_images'
model = 'ResNet50'
model_name = 'model_ResNet50_5000epochs.pth'

model_pth = f'{model_dir}/{model_name}'

#############
set_determinism(seed=0)

print_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

allimg_label = pd.read_csv(image_codebook,sep='\t')
print(allimg_label)
allimg_label[img_type] = f'{img_dir}/{allimg_label['img_name'].astype(str)}'

allimg_train = allimg_label[allimg_label['Split_datasets'] == 'Train']
allimg_val = allimg_label[allimg_label['Split_datasets'] == 'Validate']
allimg_test = allimg_label[allimg_label['Split_datasets'] == 'Test']

# train
train_x=allimg_train[img_dir].tolist()
train_y=allimg_train['caco'].tolist()

# val
val_x=allimg_val[img_dir].tolist()
val_y=allimg_val['caco'].tolist()

# test
test_x=allimg_test[img_dir].tolist()
test_y=allimg_test['caco'].tolist()

print(f'Training N= %d (Case= %d, Control= %d)'%(len(train_x),train_y.count(1),train_y.count(0)))
print('Validation N= %d (Case= %d, Control= %d)'%(len(val_x),val_y.count(1),val_y.count(0)))
print('Test N= %d (Case= %d, Control= %d)'%(len(test_x),test_y.count(1),test_y.count(0)))

val_transforms = Compose([EnsureChannelFirst(),ScaleIntensity(),Resize([600,216]), EnsureType()])
y_pred_trans = Compose([EnsureType(), Activations(softmax=True)])
y_trans = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

test_ds = ImageDataset(image_files=test_x, labels=test_y, transform = val_transforms)
test_loader = DataLoader(test_ds, batch_size=1,num_workers=2,pin_memory=torch.cuda.is_available())

if model == 'DenseNet121':
    model = DenseNet121(spatial_dims=2, in_channels=3,out_channels=2).to(device)
elif model == 'ResNet50':
    model = resnet.ResNet(block = 'bottleneck',layers = [3,4,6,3], block_inplanes = resnet.get_inplanes(),spatial_dims=2, n_input_channels=3,num_classes=2).to(device)

val_interval = 1


model.load_state_dict(torch.load(
    os.path.join(model_pth)))

#Testing data
model.eval()
test_y_pred = []
test_y_true = []
sample_lst = []
with torch.no_grad():
    test_y_value = torch.tensor([], dtype=torch.float32, device=device)
    for test_data in test_loader:
        test_images, test_labels = (test_data[0].to(device),test_data[1].to(device),)
        test_output = model(test_images)
        test_pred = test_output.argmax(dim=1)
        test_y_value = torch.cat([test_y_value, test_output], dim=0)
        sample = test_images.meta['filename_or_obj']
        for i in range(len(sample)):
            sample_lst.append(sample[i])
            test_y_true.append(test_labels[i].item())
            test_y_pred.append(test_pred[i].item())

df_test = pd.DataFrame(data={'samples': sample_lst,  'labels': test_y_true,'predicts':test_y_pred})


model_name_1 = model_name.strip('.pth')
# Compute metrics
auc = roc_auc_score(test_y_true, softmax(test_y_value.to('cpu').numpy(), axis=1)[:, 1])
acc = accuracy_score(test_y_true, test_y_pred)
sens = recall_score(test_y_true, test_y_pred)
tn, fp, fn, tp = confusion_matrix(test_y_true, test_y_pred).ravel()
spec = tn / (tn + fp)
ppv = tp / (tp + fp)
npv = tn / (tn + fn)
f1 = f1_score(test_y_true, test_y_pred)

# Save metrics to txt using pandas
new_row = pd.DataFrame({
    'Model': [model_name_1],
    'AUC': [f'{auc:.4f}'],
    'Accuracy': [f'{acc:.4f}'],
    'Sensitivity': [f'{sens:.4f}'],
    'Specificity': [f'{spec:.4f}'],
    'PPV': [f'{ppv:.4f}'],
    'NPV': [f'{npv:.4f}'],
    'F1': [f'{f1:.4f}']
})

if os.path.exists(filepath):
    df = pd.read_csv(filepath, sep='\t')
    df = pd.concat([df, new_row], ignore_index=True)
else:
    df = new_row

df.to_csv( f'../Results/Medical_images/{model_name_1}_evaluation_results.txt', sep='\t', index=False)

# Save confusion matrix as PNG
classes = ('0', '1')
cf_matrix = confusion_matrix(test_y_true, test_y_pred)
df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes)

plt.figure(figsize=(6, 3.5))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.xlabel('Predicted Values')
plt.ylabel('Truth Labels')
plt.title(f'Confusion Matrix')
plt.tight_layout()
plt.savefig(f'../Results/Medical_images/{model_name_1}_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()




