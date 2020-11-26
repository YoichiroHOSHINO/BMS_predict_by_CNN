# coding:utf-8
 
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing import image as img
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

# 学習用画像データセットのパスを指定してください。
X = np.load('./File_name_of_Img_data_for_train.npy')  
# 学習用座標データセットのパスを指定してください。
Y = np.load('./File_name_of_label_data_fotrain.npy')

# 任意の試験名を指定してください。出力結果はすべてこの名前＋αが付けられます。
testname = 'loincenter_train'

# 座標を 0 < x or y < 1 に変換します。
Y = Y.astype(np.float32)
Y[:,0:1] = Y[:,0:1]/350
Y[:,1] = Y[:,1]/250

X = X.reshape(X.shape[0], 250, 350,1)

# 学習用データセットの20%を検証群にランダム分割。
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

# バッチサイズ指定。
bat = 64

datagen = img.ImageDataGenerator(samplewise_std_normalization=True)
train_gen = datagen.flow(X_train, y_train, batch_size=bat)
test_gen = datagen.flow(X_test, y_test, batch_size=bat)

# CNNを構築
model = Sequential()
 
model.add(Conv2D(32, (5, 5), padding='same',input_shape=(250,350,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(32, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Flatten())
model.add(Dense(10000))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(2))

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.00000001, decay=0.01, amsgrad=False) 

model.summary()

# コンパイル
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])

# モデル図を出力。
plot_model(model, show_shapes=True , to_file='./plotmodel_' + testname + '.png')

# weightsフォルダにモデルを保存する。
model_checkpoint = ModelCheckpoint(filepath='./weights/model_' + testname + '.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# エポック数など指定。
history = model.fit_generator(train_gen, steps_per_epoch=len(X_train)/bat, initial_epoch=0, epochs=200, verbose=1, validation_data=test_gen, validation_steps=len(X_test)/bat, callbacks=[model_checkpoint])

# weightsフォルダに学習履歴を保存。
pd.DataFrame(history.history).to_csv('./weights/histry_' + testname + '.csv')

# 評価 & 評価結果を逐次出力
print(model.evaluate(X_test, y_test))

# グラフ描画
mae = history.history['mean_absolute_error']
val_mae = history.history['val_mean_absolute_error']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

# 平均絶対誤差グラフ
plt.plot(epochs, mae, 'b' ,label = 'training mae')
plt.plot(epochs, val_mae, 'r' , label= 'validation mae')
plt.title('Training and Validation mae')
plt.legend()

plt.figure()

# 損失グラフ
plt.plot(epochs, loss, 'b' ,label = 'training loss')
plt.plot(epochs, val_loss, 'r' , label= 'validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()

