# coding:utf-8
 
import keras
from keras.utils import np_utils
from keras.models import Sequential, Model, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image as img
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.utils import plot_model

# 予測するロース芯画像データセットのパスを指定してください。
P = np.load('./loin_img_test.npy')

# 予測する出荷年データセットのパスを指定してください。
Year = np.load('./loin_year_test.npy')

# 予測するBMSデータセット（未知の場合は仮入力で作成）のパスを指定してください。
Y = np.load('./loin_label_test.npy')

# 任意の試験名を指定してください。
testname = 'BMS_predict_withYear'


def gen_flow_for_two_inputs(datagen, batch, x_image, x_text, y_train, shuffle=True):
    # Create index
    x_index = np.arange(x_image.shape[0])
    # Pass index to the 2nd parameter instead of labels
    batch = datagen.flow(x_image, x_index, batch_size=batch, shuffle=shuffle)
    while True:
        batch_image, batch_index = batch.next()
        yield [batch_image, x_text[batch_index]], y_train[batch_index]

bat = 64

datagen = img.ImageDataGenerator()

# 学習済みモデルの読み込み。
# .hdf5ファイルのパスを指定してください。
model = load_model('./weights/model_BMS_train_withYear.hdf5')

# 予測
P = np.expand_dims(P, axis=-1)

Year = Year - 17
Year = np_utils.to_categorical(Year, 9)
Year = np.expand_dims(Year, axis=-1)

Y = Y - 2
Y = np_utils.to_categorical(Y, 11)

print ('予測します')
predict_gen_flow = gen_flow_for_two_inputs(datagen, bat, P, Year, Y, shuffle=False)
pred = model.predict_generator(predict_gen_flow, steps=len(P)/bat, verbose=1)

print ('予測値を保存します')
np.savetxt('./weights/pred_' + testname + '.csv', pred, delimiter=',')






