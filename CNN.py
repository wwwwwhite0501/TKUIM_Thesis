from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

#設定參數
img_width, img_height = 30, 30 #統一所有輸入圖片的像素大小
batch_size = 60 #每次訓練讀取的數量

#建立ImageDataGenerator #自動讀取、處理與增強資料 #將資料集分割成80%訓練組與20%驗證組
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

#訓練組
train_generator = train_datagen.flow_from_directory( #flow_from_directory根據資料夾名稱自動產生對應的標籤
    r'C:/Users/USER/OneDrive/桌面/專題/BabyCryDataset', #主資料夾
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
#驗證組
validation_generator = train_datagen.flow_from_directory(
    r'C:/Users/USER/OneDrive/桌面/專題/BabyCryDataset',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


"""
===改用ImageDataGenerator和flow_from_directory來讀取資料夾分類的圖片
不需要手動載入、打散、正規化、one-hot標籤，程式更簡潔，比較不容易出錯===


#設定 np 亂數種子
np.random.seed(10)

#載入訓練資料集
n = 10000
img_feature = np.fromfile("C:/Users/USER/OneDrive/桌面/專題/BabyCryDataset/spectrogram_images", dtype=np.uint8)
img_feature = img_feature.reshape(n, 30, 30, 3)
img_label = np.fromfile("./your/image/training/array.labels", dtype=np.uint8)
img_label = img_label.reshape(n, 1)

#打散資料集
indexs = np.random.permutation(img_label.shape[0])
rand_img_feature = img_feature[indexs]
rand_img_label = img_label[indexs]

#資料正規化
#將 feature 數字轉換為 0~1 的浮點數，能加快收斂，並提升預測準確度
#把維度 (n,30,30,3) => (n, 30*30*3)後，再除255
img_feature_normalized = rand_img_feature.reshape(n, 30*30*3).astype('float32') / 255

#將 label 轉換為 onehot 表示
img_label_onehot = to_categorical(rand_img_label)

"""


    
#建立一個線性堆疊模型
model = Sequential()

#建立第1層券積層，透過濾鏡產生32個影像特徵
model.add(Conv2D(filters=32, kernel_size=(3,3),
                 input_shape=(img_width, img_height, 3),
                 activation='relu',
                 padding='same'))

#在第1層券積層加入Dropout層，避免overfitting
model.add(Dropout(rate=0.25))

#建立第1層池化層，將32*32影像，縮小為16*16影像
model.add(MaxPooling2D(pool_size=(2,2)))

#建立第2層券積層，透過濾鏡產生64個影像特徵
model.add(Conv2D(filters=64, kernel_size=(3,3),
                 activation='relu',
                 padding='same'))

#在第2層券積層加入Dropout層，避免overfitting
model.add(Dropout(rate=0.25))

#建立第2層池化層，將16*16影像，縮小為8*8影像
model.add(MaxPooling2D(pool_size=(2,2)))

#建立平坦層，將64個8*8影像轉換為一維向量，64*8*8=4096個數字
model.add(Flatten())
model.add(Dropout(rate=0.25))

#建立有256個神經元的隱藏層 #1920~512太大
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.25))

#建立有2個神經元的輸出層
model.add(Dense(train_generator.num_classes, activation='softmax'))

#定義訓練方式
model.compile(loss='categorical_crossentropy', #損失函數
             optimizer='adam', #最佳化方法
             metrics=['accuracy']) #評估方式:準確度

#顯示模型摘要
print(model.summary())

#開始訓練模型
train_history = model.fit(train_generator,
                         epochs=55, #執行55次訓練
                         batch_size=60, #批次訓練，每批次60筆資料
                         verbose=2) #顯示訓練過程

#儲存模型
model.save("C:/Users/USER/OneDrive/桌面/專題/Model/model.keras") #.keras