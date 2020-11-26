# 枝肉断面画像（カラー）（4000x3000）から、
# ロース芯中央の座標を中心に1500x1500の正方形のロース芯画像を切り出し、
# グレースケール化して300x300にリサイズし、.bmpファイルで保存する。

import numpy as np
import pandas as pd
from PIL import Image
import sys, os
from tqdm import tqdm
import random


# 元画像フォルダを指定してください。
largeImgFolder = './Name_of_Img_folder/'

# 出力先フォルダを指定してください。
targetFolder = './Name_of_loinImg_folder/'

# 画像リストcsvファイルのパスを指定してください。
# csvファイルは「枝肉断面画像のファイル名」，「X座標の数値」，「Y座標の数値」を、
# img, center_x, center_y　のカラム名で入力しておく。
datas = pd.read_csv('./File_name_of_Img_and_center_list.csv')

for i in tqdm(datas.index):
	filename = datas.loc[i,'img']
	x = int(datas.loc[i,'center_x'])
	y = int(datas.loc[i,'center_y'])
	img = Image.open(largeImgFolder + filename)
	
	img_c = img.crop((x-750, y-750, x+749, y+749))
	img_c_gs = img_c.convert('L')
	img_c_gs_rs = img_c_gs.resize((300,300))
	
	name = filename + '.bmp'
	img_c_gs_rs.save(targetFolder + name)












