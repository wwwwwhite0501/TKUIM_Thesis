from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

import numpy as np

#設定參數
img_width, img_height = 64, 64 #統一所有輸入圖片的像素大小
batch_size = 45 #每次訓練讀取的數量

#建立ImageDataGenerator #自動讀取、處理與增強資料 #圖片隨機旋轉、縮放，讓模型更有泛化能力
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True #水平翻轉
    #zoom_range=0.05, #輕微縮放
    #rotation_range=5 #小幅旋轉
)

#訓練組
train_generator = train_datagen.flow_from_directory(
    r'C:/Users/USER/OneDrive/桌面/專題/BabyCryDataset頻譜圖/train',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)
#驗證組
validation_generator = train_datagen.flow_from_directory(
    r'C:/Users/USER/OneDrive/桌面/專題/BabyCryDataset頻譜圖/val',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)
#測試組
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    r'C:/Users/USER/OneDrive/桌面/專題/BabyCryDataset頻譜圖/test',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

#建立一個線性堆疊模型
model = Sequential()

#建立第1層券積層，透過濾鏡產生32個影像特徵
model.add(Conv2D(filters=32, kernel_size=(3,3),
                 input_shape=(img_width, img_height, 3),
                 activation='relu',
                 padding='same'))

#在第1層券積層加入Dropout層，避免overfitting
model.add(Dropout(rate=0.20))

#建立第1層池化層，將32*32影像，縮小為16*16影像
model.add(MaxPooling2D(pool_size=(2,2)))

#建立第2層券積層，透過濾鏡產生64個影像特徵
model.add(Conv2D(filters=64, kernel_size=(3,3),
                 activation='relu',
                 padding='same'))

#在第2層券積層加入Dropout層，避免overfitting
model.add(Dropout(rate=0.20))

#建立第2層池化層，將16*16影像，縮小為8*8影像
model.add(MaxPooling2D(pool_size=(2,2)))

#建立平坦層，將64個8*8影像轉換為一維向量，64*8*8=4096個數字
model.add(Flatten())
model.add(Dropout(rate=0.20))

#建立有128個神經元的隱藏層 
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.20))

#建立有2個神經元的輸出層
model.add(Dense(train_generator.num_classes, activation='softmax'))

#定義訓練方式
model.compile(loss='categorical_crossentropy', #損失函數
             optimizer='adam', #最佳化方法
             metrics=['accuracy']) #評估方式:準確度

#顯示模型摘要
print(model.summary())

#導入回調函數，避免過擬合
earlystop = EarlyStopping(
    monitor='val_loss', #監控驗證組損失
    patience=5, #5個epoch都沒進步就停止
    restore_best_weights=True) #回復最佳權重

#開始訓練模型
train_history = model.fit(train_generator,
                         epochs=75, #執行75次訓練
                         batch_size=45, #批次訓練，每批次45筆資料
                         validation_data=validation_generator, #驗證準確率與損失
                         verbose=2,
                         callbacks=[earlystop]) #顯示訓練過程

#儲存模型
model.save("C:/Users/USER/OneDrive/桌面/專題/Model/model-003.keras") #.keras

#顯示測試組準確率
print("------------------------------\n模型在20%測試組的準確率: ")
test_loss, test_acc = model.evaluate(test_generator, verbose=2)


import matplotlib.pyplot as plt

#繪製Keras訓練/驗證準確率曲線
plt.plot(train_history.history['accuracy'], label='Training Accuracy')
plt.plot(train_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(r'C:/Users/USER/OneDrive/桌面/專題/Model/learning_curve_003.png')
plt.show()