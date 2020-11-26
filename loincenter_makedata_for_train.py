#%%

# ロース芯中央予測用データセットの作成

from keras.utils import np_utils
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

# 画像リストcsvファイルの読み込み
# csvファイルは「枝肉断面画像のパス」，「X座標の数値」，「Y座標の数値」を、
# imgpath, x, y のカラム名で入力しておく。
imglist = pd.read_csv('./File_name_of_Img_list.csv') 

X = []
Y = []
for i in tqdm(imglist.index):
    imgpath = imglist.loc[i,'imgpath']
    center_x = imglist.loc[i,'x']
    center_y = imglist.loc[i,'y']
    image = Image.open(imgpath)
    data = np.asarray(image)
    X.append(data)
    Y.append([center_x, center_y])

X = np.array(X)
Y = np.array(Y)

X = X.astype('float32')
X = X / 255.0

# 
np.save('./File_name_for_Img_data.npy', X) # 学習用画像データファイル名を指定
np.save('./File_name_for_label_data.npy', Y) # 学習用座標データファイル名を指定

