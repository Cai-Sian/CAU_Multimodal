#!/bin/python3

# CNN model
# DenseNet121

import logging
import sys
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import classification_report
import monai
from monai.config import print_config
from monai.data import decollate_batch, ImageDataset,DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (EnsureChannelFirst,Activations,AddChannel,AsDiscrete,Compose,LoadImage,RandFlip,RandRotate,RandZoom,ScaleIntensity,EnsureType,Resized,Resize,RandRotate90)
from monai.utils import set_determinism
import pandas as pd
import time

print_config()

###

img_dir = '../image_inputs/preprocess'
img_type = 'preprocessed'

image_codebook = '../image_inputs/CNN_sample_split_paper.txt'

#Save model_name
max_epochs = 500
lr_initial_times = -3
model_pth=f'../Results/Medical_images/model_DenseNet121_{max_epochs}epochs'

###############
set_determinism(seed=0)
pin_memory = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print_config()

allimg_label = pd.read_csv(image_codebook,sep='\t')
print(allimg_label)
allimg_label[img_type] = f'{img_dir}/{allimg_label['img_name'].astype(str)}'

allimg_train = allimg_label[allimg_label['Split_datasets'] == 'Train']
allimg_val = allimg_label[allimg_label['Split_datasets'] == 'Validate']
allimg_test = allimg_label[allimg_label['Split_datasets'] == 'Test']

# train
train_x=allimg_train[img_type].tolist()
train_y=allimg_train['caco'].tolist()

# val
val_x=allimg_val[img_type].tolist()
val_y=allimg_val['caco'].tolist()

# test
test_x=allimg_test[img_type].tolist()
test_y=allimg_test['caco'].tolist()

print(f"Training N= %d (Case= %d, Control= %d)"%(len(train_x),train_y.count(1),train_y.count(0)))
print("Validation N= %d (Case= %d, Control= %d)"%(len(val_x),val_y.count(1),val_y.count(0)))
print("Test N= %d (Case= %d, Control= %d)"%(len(test_x),test_y.count(1),test_y.count(0)))


train_transforms = Compose([EnsureChannelFirst(),ScaleIntensity(),Resize([600,216]), RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),RandFlip(spatial_axis=0, prob=0.5),RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),EnsureType()])
val_transforms = Compose([EnsureChannelFirst(),ScaleIntensity(),Resize([600,216]), EnsureType()])
y_pred_trans = Compose([EnsureType(), Activations(softmax=True)])
y_trans = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

####

train_ds = ImageDataset(image_files=train_x, labels=train_y, transform = train_transforms)
train_loader = DataLoader(train_ds, batch_size=32,num_workers=4, shuffle=True,pin_memory=torch.cuda.is_available())

val_ds = ImageDataset(image_files=val_x, labels=val_y, transform = val_transforms)
val_loader = DataLoader(val_ds, batch_size=32,num_workers=4,pin_memory=torch.cuda.is_available())

####
model = DenseNet121(spatial_dims=2, in_channels=3,out_channels=2).to(device)

lr_rate_initial = 10**(lr_initial_times)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr = lr_rate_initial)
mode='min'
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1,mode=mode, patience=5)

###
val_interval = 1
auc_metric = ROCAUCMetric()
###
start = time.time()

best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
validate_loss_value=[]
epoch_accuracy_values = []
validate_accuracy_values = []
lr_values = []

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    step_val = 0
    y_pred_train = torch.tensor([], dtype=torch.float32, device=device)
    y_train = torch.tensor([], dtype=torch.long, device=device)
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
#         print(inputs.size())
        optimizer.zero_grad()
        outputs = model(inputs)
        y_pred_train = torch.cat([y_pred_train, outputs], dim=0)
        y_train = torch.cat([y_train, labels], dim=0)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}")
        epoch_len = len(train_ds) // train_loader.batch_size

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    acc_value_train = torch.eq(y_pred_train.argmax(dim=1), y_train)
    acc_metric_train = acc_value_train.sum().item() / len(acc_value_train)
    epoch_accuracy_values.append(acc_metric_train)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            val_epoch_loss=0
            val_correct = 0
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for val_data in val_loader:
                step_val += 1
                val_images, val_labels = (val_data[0].to(device),val_data[1].to(device),)
                validate_output = model(val_images)
                y_pred = torch.cat([y_pred, validate_output], dim=0)
                y = torch.cat([y, val_labels], dim=0)
                val_loss = loss_function(validate_output, val_labels)
                val_epoch_loss += val_loss.item()
                val_epoch_len = len(val_ds) // val_loader.batch_size

            val_epoch_loss /= step_val
            validate_loss_value.append(val_epoch_loss)
            y_onehot = [y_trans(i).to(device) for i in decollate_batch(y)]
            y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
            auc_metric(y_pred_act, y_onehot)
            result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot
            metric_values.append(result)
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            validate_accuracy_values.append(acc_metric)
            if result > best_metric:
                best_metric = result
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), model_pth + '.pth')
                print("saved new best metric model")
            print(f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                f" current accuracy: {acc_metric:.4f}"
                f" best AUC: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}")

    lr_val =  optimizer.param_groups[0]['lr']
    lr_values.append(lr_val)

    if scheduler != None:
        if mode=='min':
            print(validate_loss_value[-1])
            scheduler.step(validate_loss_value[-1])
        elif mode=='max':
            print(validate_accuracy_values[-1])
            scheduler.step(validate_accuracy_values[-1])



print(
    f"train completed, best_metric: {best_metric:.4f} "
    f"at epoch: {best_metric_epoch}")

end = time.time()
print(end - start)

result=pd.DataFrame([torch.tensor(epoch_loss_values).numpy(),torch.tensor(metric_values).numpy(),torch.tensor(validate_loss_value).numpy(),torch.tensor(epoch_accuracy_values).numpy(),torch.tensor(validate_accuracy_values).numpy(),torch.tensor(lr_values).numpy()],index=["epoch_loss_values","metric_values","validate_loss_value","epoch_accuracy_values","validate_accuracy_values","lr_values"]).T

## record the training
result.to_csv(f'{model_pth}_result_loss_metric.txt',sep='\t')
