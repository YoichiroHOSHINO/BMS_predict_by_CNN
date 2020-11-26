# coding:utf-8
 
import keras
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing import image as img
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

# 任意の試験名を指定してください。
testname = 'loincenter_predict'

bat = 64

datagen = img.ImageDataGenerator(samplewise_std_normalization=True)

# 学習済みモデルの読み込み。
# .hdf5ファイルのパスを指定してください。
model = load_model('./weights/model_loincenter_train.hdf5')

print ('予測用画像を読み込みます')
# 予測する画像データセットのパスを指定してください。
P = np.load('./gifu_H19-H25_loincenter_withdata_img.npy')

print ('予測用画像を変換します')
P = P.reshape(P.shape[0], 250, 350,1)

print ('予測します')
predict_gen = datagen.flow(P, shuffle=None, batch_size=bat)
pred = model.predict_generator(predict_gen, steps=len(predict_gen), verbose=1)

print ('予測値を保存します')
# weightsフォルダに予測結果が保存されます。
np.savetxt('./weights/pred_' + testname + '.csv', pred, delimiter=',')

