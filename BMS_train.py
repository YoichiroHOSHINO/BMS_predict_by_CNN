# coding:utf-8
 
import keras
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image as img
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.utils import plot_model

# 訓練群、検証群データセットのパスを指定してください。
X_train = np.load('./loin_img_train.npy')
Y_train = np.load('./loin_label_train.npy')
X_test = np.load('./loin_img_test.npy')
Y_test = np.load('./loin_label_test.npy')

# 任意の試験名を指定してください。出力結果はすべてこの名前＋αが付けられます。
testname = 'BMS_train'

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# BMSをカテゴリカルデータに変換。BMS1がないので、BMS2-12を0-10の11カテゴリにする。
Y_train = Y_train - 2
Y_train = np_utils.to_categorical(Y_train, 11)
Y_test = Y_test - 2
Y_test = np_utils.to_categorical(Y_test, 11)

# バッチサイズ、画像前処理の指定。
bat = 64
datagen = img.ImageDataGenerator(rotation_range=30, height_shift_range=0.1, width_shift_range=0.1)
train_gen = datagen.flow(X_train, Y_train, batch_size=bat)
test_gen = datagen.flow(X_test, Y_test, batch_size=bat)

# CNNを構築
inputs = Input(shape=(300,300,1))

x = BatchNormalization(axis=-1, epsilon=0.001)(inputs)
x = Conv2D(32, (5, 5), padding='same', name='conv_1')(x)
x = Activation('relu', name='relu_1')(x)
x = MaxPooling2D(pool_size=(3, 3), name='pool_1')(x)
x = Dropout(0.05)(x)
x = Conv2D(32, (5, 5), padding='same', name='conv_2')(x)
x = Activation('relu', name='relu_2')(x)
x = MaxPooling2D(pool_size=(3, 3), name='pool_2')(x)
x = Conv2D(32, (5, 5), padding='same', name='conv_3')(x)
x = Activation('relu', name='relu_3')(x)
x = MaxPooling2D(pool_size=(3, 3), name='pool_3')(x)

x = Flatten()(x)
x = Dense(10000, activation='relu', name='dense_relu_1')(x)
x = Dropout(0.05)(x)
x = Dense(1000, activation='relu', name='dense_relu_2')(x)
x = Dropout(0.05)(x)
x = Dense(100, activation='relu', name='dense_relu_3')(x)
predictions = Dense(11, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.00000001, decay=0.001, amsgrad=False) 
model.compile(optimizer=adam, loss='categorical_crossentropy',  metrics=['accuracy'])

# モデル図を出力。
plot_model(model, show_shapes=True , to_file='./plotmodel_' + testname + '.png')

# weightsフォルダにモデルを保存する。
model_checkpoint = ModelCheckpoint(filepath='./weights/model_' + testname + '.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# エポック数など指定。
history = model.fit_generator(train_gen, steps_per_epoch=len(X_train)/bat, initial_epoch=1, epochs=200, verbose=1, validation_data=test_gen, validation_steps=len(X_test)/bat, callbacks=[model_checkpoint])

# weightsフォルダに学習履歴を保存。
pd.DataFrame(history.history).to_csv('./weights/histry_' + testname + '.csv')

# 評価 & 評価結果を逐次出力
print(model.evaluate(X_test, Y_test))

# 予測
print ('予測用画像を読み込みます')
P = np.load('./loin_img_test.npy')

print ('予測用画像を変換します')
P = np.expand_dims(P, axis=-1)

print ('予測します')
predict_gen = datagen.flow(P, shuffle=None, batch_size=bat)
pred = model.predict_generator(predict_gen, steps=len(predict_gen), verbose=1)

print ('予測値を保存します')
np.savetxt('./weights/pred_' + testname + '.csv', pred, delimiter=',')

# グラフ描画
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

# 1) 正確度グラフ
plt.plot(epochs, acc, 'b' ,label = 'training acc')
plt.plot(epochs, val_acc, 'r' , label= 'validation acc')
plt.title('Training and Validation acc')
plt.legend()

plt.figure()

# 損失グラフ
plt.plot(epochs, loss, 'b' ,label = 'training loss')
plt.plot(epochs, val_loss, 'r' , label= 'validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()




