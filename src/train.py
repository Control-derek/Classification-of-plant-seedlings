import os
import cv2
from PIL import Image
import numpy as np
import pickle as pkl
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from processer import equalize, zmIceColor, extractGreen
from dataset import PlantSeedDataset
from xgboost import XGBClassifier
import xgboost
from sklearn.model_selection import train_test_split

# 准备数据集
datasets_path = "../dataset/classifer/train"
    
transform = transforms.Compose([
        transforms.Resize((256, 256)),
        np.asarray,
        transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2BGR))
        ])
dataset = PlantSeedDataset(datasets_path, transform=transform)

with open('hog_features.pkl', 'rb') as f:
    hog_features = pkl.load(f)
    
X_train, X_test, y_train, y_test =train_test_split(hog_features, dataset.labels, test_size=0.2, random_state=0)

model_xgb = XGBClassifier(learning_rate=0.05, 
                          objective='binary:logistic',
                          booster="gbtree",
                          num_class=12, 
                          n_estimators=1000,
                          tree_method="hist",
                          device="cuda",
                          use_label_encoder=False)
# model_xgb.fit(X_train, y_train)

model_xgb.fit(X_train, 
              y_train, 
              early_stopping_rounds=10, 
              eval_set=[(X_test, y_test)], 
              eval_metric='mlogloss', 
              verbose=50)

model_xgb.save_model("xgb_hog.json")

import sklearn.metrics
y_pred = model_xgb.predict(X_test)
sklearn.metrics.f1_score(y_test, y_pred, average='micro')