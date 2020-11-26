#%%

# BMS学習用データの作成。訓練群と検証群に分けて作成する。

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

# 画像リストcsvファイルのパスを指定してください。
# 画像はあらかじめ、訓練群（train）か検証群（test）かを決定しておく。
# csvファイルは「ロース芯画像のファイル名」，「BMS」，「出荷年」,「train/test」を、
# img, BMS, year, class のカラム名で入力しておく。
imglist = pd.read_csv('./File_name_of_loinImg_list.csv')

# ロース芯画像フォルダのパスを指定してください。
imgfolder = './Name_of_loinImg_folder/'

X_train = []
X2_train = []
Y_train = []
X_test = []
X2_test = []
Y_test = []

for i in tqdm(imglist.index):
    img = Image.open(imgfolder + imglist.loc[i, 'img'])
    bms = imglist.loc[i, 'BMS']
    year = imglist.loc[i, 'year']
    sep = imglist.loc[i, 'class']
    data = np.asarray(img)
    if sep == 'train':
        X_train.append(data)
        X2_train.append(year)
        Y_train.append(bms)
    else:
        X_test.append(data)
        X2_test.append(year)
        Y_test.append(bms)

X_train = np.array(X_train)
X2_train = np.array(X2_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
X2_test = np.array(X2_test)
Y_test = np.array(Y_test)

X_train = X_train.astype('float32')
X_train = X_train / 255.0
X_test = X_test.astype('float32')
X_test = X_test / 255.0

np.save('./loin_img_train.npy', X_train)
np.save('./loin_year_train.npy', X2_train)
np.save('./loin_label_train.npy', Y_train)
np.save('./loin_img_test.npy', X_test)
np.save('./loin_year_test.npy', X2_test)
np.save('./loin_label_test.npy', Y_test)

