#%%

# ロース芯中央を予測したい枝肉断面画像を特定フォルダに入れておき、
# 予測用データセットと画像リストcsvファイルを作成する。


from keras.utils import np_utils
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

# ロース芯中央を予測する画像が入っているフォルダ名を指定してください。
folder = './Name_of_Img_folder/'
files = os.listdir(folder)

X = []
filelist = []
for f in tqdm(files):
    imgpath = folder + f
    image = Image.open(imgpath)
    data = np.asarray(image)
    X.append(data)
    filelist.append([f])

X = np.array(X)
filelist = np.array(filelist)

X = X.astype('float32')
X = X / 255.0

# 予測する画像データセットの出力名を指定してください。
np.save('./File_name_of_Img_data_for_predict.npy', X)
# 画像データリストのcsvファイル名を指定してください。
np.savetxt('./File_name_of_Img_list_for_predict.csv', filelist, delimiter=',', fmt='%s')

