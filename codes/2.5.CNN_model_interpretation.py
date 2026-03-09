##### GradCAM ##### 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad,LayerCAM,DeepFeatureFactorization
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from torchvision import transforms
import cv2
import os
from monai.data import ImageDataset
from monai.networks.nets import resnet
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    ScaleIntensity,
    EnsureType,
    Resize,
)

########
img_dir = '../image_inputs/preprocess'
img_type = 'preprocessed'
test_set_codebook = '../image_inputs/testing_result_summary.txt' ##index for the testing samples

model_dir = '../Results/Medical_images'
model = 'ResNet50'
model_name = 'model_ResNet50_5000epochs.pth'
target = 'layer4_final'
model_pth = f'{model_dir}/{model_name}'

Cam_model = 'GradCAMplus'

#save XAI results
model_name_1 = model_name.strip('.pth')
directory = f'../Results/Medical_images/XAI_{Cam_model}/{model_name_1}/{target}'
os.makedirs(directory, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df_test = pd.read_csv(test_set_codebook,sep='\t')
df_truth = df_test.copy()

test_x = list(df_truth['samples'])
test_y = list(df_truth['labels'])
predic_result = list(df_truth['predicts'])


model = resnet.ResNet(block = 'bottleneck',layers = [3,4,6,3], block_inplanes = resnet.get_inplanes(),spatial_dims=2, n_input_channels=3,num_classes=2).to(device)
model.load_state_dict(torch.load(os.path.join(model_pth)))
model.eval()


cam_plot = GradCAMPlusPlus(model=model, target_layers=[model.layer4[-1]])
    
for sample in np.unique(test_x):
    index_temp=pd.Series(test_x)==sample
    cam_x=pd.Series(test_x)[index_temp].tolist()
    print(cam_x)
    cam_y=pd.Series(test_y)[index_temp].tolist()
    pred_y=pd.Series(predic_result)[index_temp].tolist()
    original = cv2.imread(cam_x[0])
    y,x,_ = original.shape
    val_transforms = Compose([EnsureChannelFirst(),ScaleIntensity(),Resize([x,y]), EnsureType()])
    cam_ds = ImageDataset(image_files=f'{img_dir}/{cam_x}', labels=cam_y, transform = val_transforms)
    cam_loader = torch.utils.data.DataLoader(cam_ds, batch_size=1,shuffle=False,num_workers=2,pin_memory=torch.cuda.is_available())
    img_case=pd.Series(test_x).tolist()
    print(img_case)
    

    i=0
    for cam_data in cam_loader:
        fig, axs = plt.subplots()
        sample_name = img_case[i].replace('.jpg','')
        plt.xlabel(f"{sample_name}; label:{cam_y[i]}; pred:{pred_y[i]}")
        axs.imshow(np.transpose(cam_data[0][0,0,:,:]),cmap = 'gray')
        plt.tight_layout()
        plt.show()
        plt.close()
        
        fig, axs = plt.subplots()
        temp_1 = np.transpose(cam_plot(cam_data[0][0:,0:,0:,0:]))
        axs.imshow(np.transpose(cam_data[0][0,0,0:,0:]),cmap = 'gray', alpha=0.7)
        im = axs.imshow(temp_1, cmap='jet',alpha=0.2)
        axs.set_xlabel("layer4.2")

        plt.tight_layout()
        plt.show()
        plt.close()

        np.save(f'{directory}/{sample_name}.npy', temp_1) # save in numpy for quantify analysis

        torch.cuda.empty_cache()
