# BMS_predict_by_CNN
# 畳み込みニューラルネットワークを用いた黒毛和種枝肉断面画像のBMS判定

スクリプトはすべてPython3で記述しています。

Keras (TensorFlow), numpy, pandas, matplotlib, sklearn, PIL 等のパッケージを利用します。

Anaconda3 をインストールすれば、Keras (TensorFlow) 以外はデフォルトでインストールされると思います。


## 枝肉断面画像からのロース芯画像の切り出し

枝肉断面画像のロース芯中央座標を予測し、ロース芯を中心とした正方形の画像を切り出す。

Pythonスクリプトを任意のフォルダに置き、直下のフォルダに枝肉断面画像をまとめて入れておきます。

### データセットの作成

　ロース芯中央座標が明らかな画像リストのcsvファイルを作成してください。

　3カラムで、1行目のカラム名をimgpath, x, yとし、それぞれに「枝肉断面画像のパス」，「X座標の数値」，「Y座標の数値」を入力したcsvファイルにしてください。

　loincenter_makedata_for_train.py に画像リストCSVファイルのパスを指定し、実行してください。

　学習用の画像データセットファイルと、座標データセットファイル（.npy）が作成されます。

### ロース芯中心座標の学習

　loincenter_train.py の学習用画像データセット、学習用座標データセットのパスを指定し、任意の試験名を指定して実行してください。

　以下のファイルが生成されます。

　同じフォルダ内　plotmodel_試験名.png

　weightsフォルダ内

- モデルファイル（学習結果）　model_試験名.hdf5
- 学習履歴　histry_試験名.csv


### 予測用データセットの作成

　loincenter_makedata_for_predict.py に、ロース芯中央を予測する画像を入れたフォルダ名、および出力する画像データセット名と画像リストのファイル名を指定して実行してください。

　
### ロース芯中心座標の予測

　loincenter_predict.py に、任意の試験名を指定し、学習済みモデルファイルのパスを指定して実行してください。

　学習済みモデルは、上の学習で出力された model_ooooo.hdf5 ファイルです。

　weightsフォルダ内に、pred_試験名.csvファイルとして予測結果が出力されます。

### ロース芯画像の切り出し

　画像リストのcsvファイルを作成しておいてください。
csvファイルの内容は「枝肉断面画像のファイル名」，「X座標の数値」，「Y座標の数値」を、img, center_x, center_y　のカラム名で入力してください。

　loin_crop_fromImg.py に、元画像（枝肉断面画像）を入れたフォルダのパスと、ロース芯画像の出力先フォルダのパス、および画像リストcsvファイルのパスを指定して実行してください。


## ロース芯画像のBMS予測

データセットの作成

手書き最高





　
