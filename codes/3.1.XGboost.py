##### XGboost #####

import sys
import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from os import listdir
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import (
    classification_report, balanced_accuracy_score, 
    roc_auc_score,roc_curve,recall_score,
    accuracy_score,precision_score,f1_score,
    confusion_matrix,average_precision_score)

from sklearn.linear_model import LogisticRegression
import math
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

###
def AUC(model, inpu, y, data):
    print("%s Reslut:" %data)
    pred=xgb_model1.predict(inpu)
    print(f"AUC: {roc_auc_score(y,pred):.4f}\n")
    return data,roc_auc_score(y,pred)


def prob_threshold(model,inpu,y):
    #y_pred_value_soft=softmax(v, axis=1)[:,1]
    allthreshold=np.arange(0,1.0, 0.01)
    threshold = []
    sensitivity = []
    specificity = []
    bind=[]
    for p in allthreshold:
        threshold.append(p)
        y_pred = (model.predict(inpu) >= p).astype(int)
        tn, fp, fn, tp = confusion_matrix(y,  y_pred).ravel()
        sensitivity.append(recall_score(y,y_pred))
        specificity.append((tn / (tn+fp)))
        bind.append(recall_score(y,y_pred)+(tn / (tn+fp)))
    return threshold[np.argmax(bind)]

def score(model, inpu,y,threshold_final,data):
    print("%s Reslut:" %data)
    y_pred = (model.predict(inpu) >= threshold_final).astype(int)
    print(f"Accuracy: {accuracy_score(y,y_pred):.4f}")
    print(f"Sensitivity: {recall_score(y,y_pred):.4f}")
    tn, fp, fn, tp = confusion_matrix(y,y_pred).ravel()
    specificity = tn / (tn+fp)
    print(f"Specificity: {specificity:.4f}")
    F1=f1_score(y,y_pred)
    print(f"F1:{F1:.4f}")
    print(f"Confusion Matrix:\n {confusion_matrix(y,y_pred)}\n")
    return accuracy_score(y,y_pred),recall_score(y,y_pred),confusion_matrix(y,y_pred),specificity,F1

###

img_featuremap = '../multimodal_inputs/model_ResNet50_5000epochs_extract_feature.txt'
demo_file = '../multimodal_inputs/TWB_demo.txt'
PRS_file = '../multimodal_inputs/PRS_value.txt'

save_dir = '../for_paper_CAU1/Results/Mulimodal'

model_name = 'Multimodal_xgb'

img_inputs = pd.read_csv(img_featuremap,sep='\t')
demo_inputs =  pd.read_csv(demo_file,sep='\t')
genetic_inputs = pd.read_csv(PRS_file,sep='\t')


allimg_label = pd.read_csv('../image_inputs/CNN_sample_split_paper.txt',sep='\t') 


df_MI_merge = demo_inputs.merge(img_inputs,on = 'MI_ID',how='right').drop_duplicates()
df_MI_merge = df_MI_merge.merge(genetic_inputs,on='Release_No',how='right').drop_duplicates()




allimg_train = allimg_label[allimg_label['Split_datasets'] == 'Train']
allimg_val = allimg_label[allimg_label['Split_datasets'] == 'Validate']
allimg_test = allimg_label[allimg_label['Split_datasets'] == 'Test']

# train
train_img_x=allimg_train['img_name'].tolist()
train_img_y=allimg_train['caco'].tolist()


# val
val_img_x=allimg_val['img_name'].tolist()
val_img_y=allimg_val['caco'].tolist()

# test
test_img_x=allimg_test['img_name'].tolist()
test_img_y=allimg_test['caco'].tolist()

print('Image-based counts')
print(f"Training N= %d (Case= %d, Control= %d)"%(len(train_img_x),train_img_y.count(1),train_img_y.count(0)))
print("Validation N= %d (Case= %d, Control= %d)"%(len(val_img_x),val_img_y.count(1),val_img_y.count(0)))
print("Test N= %d (Case= %d, Control= %d)"%(len(test_img_x),test_img_y.count(1),test_img_y.count(0)))


PRS = ['PGS_integ']

df_MI_merge_1 = df_MI_merge.merge(allimg_label[['img_name','caco']].rename(columns={'img_name':'sample_name'}),how = 'right')
df_MI_merge_1 = df_MI_merge_1.replace({'SEX':{1:0,2:1}})
df_MI_merge_1['SEX'] = df_MI_merge_1['SEX'].astype(int,errors='ignore')
df_MI_merge_1['SEX'] = df_MI_merge_1['SEX'].astype('category')

## for non-img XGBOOST (individual-based)
df_MI_merge_no_img = df_MI_merge_1[['Release_No','MI_ID','AGE','SEX','BMI','caco']+PRS].dropna(subset = 'Release_No')
df_MI_merge_no_img = df_MI_merge_no_img.drop_duplicates()


img_val = allimg_val['img_name'].str.split('_',expand=True).drop_duplicates(subset = [0])
img_test = allimg_test['img_name'].str.split('_',expand=True).drop_duplicates(subset = [0])


train = df_MI_merge_no_img[~(df_MI_merge_no_img['MI_ID'].isin(img_test[0]))]
train = train[~(train['MI_ID'].isin(img_val[0]))]

val = df_MI_merge_no_img[df_MI_merge_no_img['MI_ID'].isin(img_val[0])]

test = df_MI_merge_no_img[df_MI_merge_no_img['MI_ID'].isin(img_test[0])]
test = test[~test['MI_ID'].isin(img_val[0])]

train_x=train['MI_ID'].tolist()
train_y=train['caco'].tolist()

val_x=val['MI_ID'].tolist()
val_y=val['caco'].tolist()

test_x=test['MI_ID'].tolist()
test_y=test['caco'].tolist()

print('Sample-based counts')
print(f"Training N= %d (Case= %d, Control= %d)"%(len(train_x),train_y.count(1),train_y.count(0)))
print("Validation N= %d (Case= %d, Control= %d)"%(len(val_x),val_y.count(1),val_y.count(0)))
print("Test N= %d (Case= %d, Control= %d)"%(len(test_x),test_y.count(1),test_y.count(0)))


train_dataset = df_MI_merge_no_img[df_MI_merge_no_img['MI_ID'].isin(train_x)]
val_dataset = df_MI_merge_no_img[df_MI_merge_no_img['MI_ID'].isin(val_x)]
test_dataset = df_MI_merge_no_img[df_MI_merge_no_img['MI_ID'].isin(test_x)]

record = pd.DataFrame()
test_dataset_outcome = test_dataset.copy()
test_dataset_outcome['set'] = 'test'

train_dataset_outcome = train_dataset.copy()
train_dataset_outcome['set'] = 'train'

val_dataset_outcome = val_dataset.copy()
val_dataset_outcome['set'] = 'validate'
print(PRS)


for i in ['PGS', 'DEM', 'PGS_DEM']:
  
    if i == 'PGS':
        test_opt = PRS
    elif i == 'DEM':
        test_opt = ['AGE','SEX','BMI']
    elif i == 'PGS+DEM':
        test_opt = PRS+['AGE','SEX','BMI'] 
        
    train_dataset.dropna(subset='Release_No',inplace=True)
    train_dataset.dropna(subset=['BMI'],inplace=True)
    train_dataset.dropna(subset=PRS,inplace=True)

    traindata_x = train_dataset[test_opt]
    traindata_y=train_dataset['caco']


    val_dataset.dropna(subset='Release_No',inplace=True)
    val_dataset.dropna(subset=['BMI'],inplace=True)
    val_dataset.dropna(subset=PRS,inplace=True)

    validdata_x = val_dataset[test_opt]
    validdata_y=val_dataset['caco']

    test_dataset.dropna(subset='Release_No',inplace=True)
    test_dataset.dropna(subset=['BMI'],inplace=True)
    test_dataset.dropna(subset=PRS,inplace=True)

    testdata_x = test_dataset[test_opt]
    testdata_y=test_dataset['caco']
    
    print('train number:', len(traindata_y),'; control:',list(traindata_y).count(0),'case:',list(traindata_y).count(1))
    print('validate number:', len(validdata_y),'; control:',list(validdata_y).count(0),'case:',list(validdata_y).count(1))
    print('test number:', len(testdata_y),'; control:',list(testdata_y).count(0),'case:',list(testdata_y).count(1))
    
    dtrain = xgb.DMatrix(traindata_x, label=traindata_y, enable_categorical=True)
    dvalid = xgb.DMatrix(validdata_x, label=validdata_y, enable_categorical=True)
    dtest = xgb.DMatrix(testdata_x, label=testdata_y, enable_categorical=True)
    
    param = {}
    param['objective'] = 'binary:logistic'
    param['eval_metric'] = 'auc'
    param['seed'] = 1000
    res = {}
    num_round = 1000

    xgb_model1 = xgb.train(param, dtrain,num_round,
                           evals=[(dtrain, "train"), (dvalid, "valid")], 
                           early_stopping_rounds=30, 
                           evals_result=res)

    xgb_model1 = xgb.train(param, dtrain, 
                           xgb_model1.best_ntree_limit, 
                           evals=[(dtrain, "train"), (dvalid, "valid")], 
                           evals_result=res)
    ## save model
    xgb_model1.save_model(f'{save_dir}/{model_name}_IndBased_{i}.json')
    
    ########
    txt_tr, AUC_tr = AUC(xgb_model1,dtrain,traindata_y, 'train')
    txt_v, AUC_v =AUC(xgb_model1,dvalid,validdata_y, 'valid')
    txt_t, AUC_t =AUC(xgb_model1,dtest,testdata_y, 'test')

    threhold_select_youden_tr=prob_threshold(xgb_model1 ,dtrain,traindata_y)
    acc_tr,sen_tr,con_tr,spe_tr, F1_tr = score(xgb_model1, dtrain,traindata_y,threhold_select_youden_tr,'train')

    threhold_select_youden_v=prob_threshold(xgb_model1 ,dvalid,validdata_y)
    acc_v,sen_v,con_v,spe_v, F1_v =score(xgb_model1, dvalid,validdata_y,threhold_select_youden_v,'valid')
    acc_t,sen_t,con_t,spe_t, F1_t =score(xgb_model1, dtest,testdata_y,threhold_select_youden_v,'test')
    
    out_value_tr=xgb_model1.predict(dtrain)
    out_data_tr=pd.DataFrame({f'{i}_xgb':out_value_tr},index=traindata_y.index)
    out_value_v=xgb_model1.predict(dvalid)
    out_data_v=pd.DataFrame({f'{i}_xgb':out_value_v},index=validdata_y.index)    
    out_value=xgb_model1.predict(dtest)
    out_data=pd.DataFrame({f'{i}_xgb':out_value},index=testdata_y.index)
    
    d = {'dataset': ['Train','Validation','Test'], 'Total': [len(traindata_y), len(validdata_y),len(testdata_y)],
     'Case':[list(traindata_y).count(1),list(validdata_y).count(1),list(testdata_y).count(1)],
     'Control':[list(traindata_y).count(0),list(validdata_y).count(0),list(testdata_y).count(0)],
    'AUC':[AUC_tr,AUC_v,AUC_t],'Accuracy':[acc_tr,acc_v,acc_t],'Sensitivity':[sen_tr,sen_v,sen_t],
    'Confusion_Matrix':[con_tr,con_v,con_t],'Specificity':[spe_tr,spe_v,spe_t],'F1':[F1_tr,F1_v,F1_t],'set':[i,i,i]}
    
    record = pd.concat([record,pd.DataFrame(d)])
    test_dataset_outcome = pd.concat([test_dataset_outcome,out_data],axis=1)
    val_dataset_outcome = pd.concat([val_dataset_outcome,out_data_v],axis=1)
    train_dataset_outcome = pd.concat([train_dataset_outcome,out_data_tr],axis=1)

all_outcome = pd.concat([test_dataset_outcome,val_dataset_outcome,train_dataset_outcome],axis=0)

# save results
record.to_csv(f'{save_dir}/{model_name}_evaluation_non_img_xgb_results.txt',sep = '\t',index = False)
all_outcome.to_csv(f'{save_dir}/{model_name}_evaluation_non_img_xgb_result_score.txt',sep = '\t',index = False)



## for img XGBOOST (Image-based)

train_dataset = df_MI_merge_1[df_MI_merge_1['sample_name'].isin(train_img_x)]
val_dataset = df_MI_merge_1[df_MI_merge_1['sample_name'].isin(val_img_x)]
test_dataset = df_MI_merge_1[df_MI_merge_1['sample_name'].isin(test_img_x)]

record = pd.DataFrame()
test_dataset_outcome = test_dataset.copy()
test_dataset_outcome['set'] = 'test'

train_dataset_outcome = train_dataset.copy()
train_dataset_outcome['set'] = 'train'

val_dataset_outcome = val_dataset.copy()
val_dataset_outcome['set'] = 'validate'

CNN_model = 'model_ResNet50_500epochs'
PGS_lst = PRS
print(PGS_lst)

for i in ['PGS_IMG', 'DEM_IMG', 'IMG','DEM_IMG_PGS']:
    feature_lst = list(filter(lambda x: x.startswith('V'), df_MI_merge_1.columns))

    if i == 'PGS+IMG':
        test_opt = feature_lst+PGS_lst
    elif i == 'DEM+IMG':
        test_opt = feature_lst + ['AGE','SEX','BMI']
    elif i == 'IMG':
        test_opt =feature_lst
    elif i == 'DEM+IMG+PGS':
        test_opt = feature_lst+PGS_lst+['AGE','SEX','BMI']


    train_dataset.dropna(subset='MI_ID',inplace=True)
    train_dataset.dropna(subset='Release_No',inplace=True)
    train_dataset.dropna(subset='BMI',inplace=True)
    train_dataset.dropna(subset=PGS_lst,inplace=True)


    traindata_x = train_dataset[test_opt]
    traindata_y=train_dataset['caco']


    val_dataset.dropna(subset='MI_ID',inplace=True)
    val_dataset.dropna(subset='Release_No',inplace=True)
    val_dataset.dropna(subset='BMI',inplace=True)
    val_dataset.dropna(subset=PGS_lst,inplace=True)


    validdata_x = val_dataset[test_opt]
    validdata_y=val_dataset['caco']

    test_dataset.dropna(subset='MI_ID',inplace=True)
    test_dataset.dropna(subset='Release_No',inplace=True)
    test_dataset.dropna(subset='BMI',inplace=True)
    test_dataset.dropna(subset=PGS_lst,inplace=True)


    testdata_x = test_dataset[test_opt] 
    testdata_y=test_dataset['caco']
    
    
    dtrain = xgb.DMatrix(traindata_x, label=traindata_y, enable_categorical=True)
    dvalid = xgb.DMatrix(validdata_x, label=validdata_y, enable_categorical=True)
    dtest = xgb.DMatrix(testdata_x, label=testdata_y, enable_categorical=True)

    param = {}
    param['objective'] = 'binary:logistic'
    param['eval_metric'] = 'auc'
    param['seed'] = 1000
    res = {}
    num_round = 1000

    xgb_model1 = xgb.train(param, dtrain,num_round,
                           evals=[(dtrain, "train"), (dvalid, "valid")], 
                           early_stopping_rounds=10, 
                           evals_result=res)

    xgb_model1 = xgb.train(param, dtrain, 
                           xgb_model1.best_ntree_limit, 
                           evals=[(dtrain, "train"), (dvalid, "valid")], 
                           evals_result=res)
    ## save model
    xgb_model1.save_model(f'{save_dir}/{model_name}_ImgBased_{i}.json')
    ###########
    
    txt_tr, AUC_tr = AUC(xgb_model1,dtrain,traindata_y, 'train')
    txt_v, AUC_v =AUC(xgb_model1,dvalid,validdata_y, 'valid')
    txt_t, AUC_t =AUC(xgb_model1,dtest,testdata_y, 'test')

    threhold_select_youden_tr=prob_threshold(xgb_model1 ,dtrain,traindata_y)
    acc_tr,sen_tr,con_tr,spe_tr, F1_tr = score(xgb_model1, dtrain,traindata_y,threhold_select_youden_tr,'train')

    threhold_select_youden_v=prob_threshold(xgb_model1 ,dvalid,validdata_y)
    acc_v,sen_v,con_v,spe_v, F1_v =score(xgb_model1, dvalid,validdata_y,threhold_select_youden_v,'valid')
    acc_t,sen_t,con_t,spe_t, F1_t =score(xgb_model1, dtest,testdata_y,threhold_select_youden_v,'test')
    
    out_value_tr=xgb_model1.predict(dtrain)
    out_data_tr=pd.DataFrame({f'{i}_xgb':out_value_tr},index=traindata_y.index)    
    out_value_v=xgb_model1.predict(dvalid)
    out_data_v=pd.DataFrame({f'{i}_xgb':out_value_v},index=validdata_y.index)    
    out_value=xgb_model1.predict(dtest)
    out_data=pd.DataFrame({f'{i}_xgb':out_value},index=testdata_y.index)    

    
    d = {'dataset': ['Train','Validation','Test'], 'Total': [len(traindata_y), len(validdata_y),len(testdata_y)],
     'Case':[list(traindata_y).count(1),list(validdata_y).count(1),list(testdata_y).count(1)],
     'Control':[list(traindata_y).count(0),list(validdata_y).count(0),list(testdata_y).count(0)],
    'AUC':[AUC_tr,AUC_v,AUC_t],'Accuracy':[acc_tr,acc_v,acc_t],'Sensitivity':[sen_tr,sen_v,sen_t],
    'Confusion_Matrix':[con_tr,con_v,con_t],'Specificity':[spe_tr,spe_v,spe_t],'F1':[F1_tr,F1_v,F1_t],'set':[i,i,i]}
    
    record = pd.concat([record,pd.DataFrame(d)])
    test_dataset_outcome = pd.concat([test_dataset_outcome,out_data],axis=1)
    val_dataset_outcome = pd.concat([val_dataset_outcome,out_data_v],axis=1)
    train_dataset_outcome = pd.concat([train_dataset_outcome,out_data_tr],axis=1)

all_outcome = pd.concat([test_dataset_outcome,val_dataset_outcome,train_dataset_outcome],axis=0)

record.to_csv(f'{save_dir}/{model_name}_evaluation_{CNN_model}_xgb_results.txt',sep = '\t',index = False)
all_outcome.to_csv(f'{save_dir}/{model_name}_evaluation_{CNN_model}_xgb_result_score.txt',sep = '\t',index = False)
