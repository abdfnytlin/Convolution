## 建構預測模型
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical

# 指定亂數種子
seed = 7
np.random.seed(seed)
# 匯入資料集
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()

## 資料預處理
# 圖片轉成 4維張量
X_test = X_test.reshape(X_test.shape[0],28,28,1).astype('float32')
 # 正規化，使之介於0~1之間
X_test = X_test / 255  
# One-Hot encoding
Y_test_bk = Y_test.copy() # 備分資料集 Y_test
Y_test = to_categorical(Y_test)

# 定義模型(載入已訓練好的模型)
model = Sequential()
model = load_model("./data/mnist.h5")
# 編譯模型
model.compile(loss="categorical_crossentropy",optimizer="adam",
             metrics=["accuracy"])
# 預測模型
Y_pred = model.predict_classes(X_test)

# 測試集的長度
test_length = len(X_test)

# 找出分類錯誤的數字編號
num_err = [i for i in range(test_length) if (Y_pred[i]!=Y_test_bk[i])]

# 每列兩個數字，總列數為 ceil(分類錯誤的數字的數量/2)
cols = 2
mods = len(num_err) % 2
if mods == 1: rows = int(len(num_err)/2)+1
else: rows = int(len(num_err)/2)

# 計算每個數字是在左邊還是右邊(由索引的餘數判斷)，存成字典
num_err_col = {}
for j, x in enumerate(num_err):
    num_err_col[x] = j % 2

import matplotlib.pyplot as plt
%matplotlib inline

# 找出分類錯誤的數字有哪些，取出它們的編號
num_err = [j for j in range(test_length) if (Y_pred[j]!=Y_test_bk[j])]
# 每列兩個數字(cols)，從分類錯誤的數字的數量來計算總列數(rows)
cols = 2
mods = len(num_err) % 2
if mods == 1: rows = int(len(num_err)/2)+1
else: rows = int(len(num_err)/2)

# 計算每個數字是在左邊還是右邊(由索引的餘數判斷)，存成字典
num_err_col = {}
for j, x in enumerate(num_err):
    num_err_col[x] = j % 2
# print(num_err_col)
    
# 指定亂數種子
seed = 7
np.random.seed(seed)
# 匯入資料集
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()

row = 0
for x in num_err:
# 預測錯誤的圖像編號
    i = x
# 記錄圖像資料
    digit = X_test[i].reshape(28,28)
# 轉成 4D 張量
    X_test_digit = X_test[i].reshape(1,28,28,1).astype("float32")
    X_test_digit = X_test_digit / 255
# 繪製圖片
    if num_err_col[i] == 0:
        plt.figure(figsize=[15,4])
        plt.subplot(1,4,1)
    if num_err_col[i] == 1:
        plt.subplot(1,4,3)
    plt.title("Example of Digit: " + str(Y_test_bk[i]))
    plt.imshow(digit, cmap="gray")
# 計算機率
    probs = model.predict_proba(X_test_digit, batch_size=1)
    # print(probs)
# 繪製機率分布圖
    if num_err_col[i] == 0:
        plt.subplot(1,4,2)
    if num_err_col[i] == 1:
        plt.subplot(1,4,4)
    plt.title("Prob of the Digit")
    plt.bar(np.arange(10),probs.reshape(10), align="center")
    plt.xticks(np.arange(10),np.arange(10).astype(str))
    
# 換行，到最後一行時停止
    if num_err_col[i] == 1:
        plt.show()
        row = row + 1
    if row >= rows:
        break
