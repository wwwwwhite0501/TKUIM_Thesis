from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np
import os
from datetime import datetime

#自動設定路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

#數據路徑
data_base_path = os.path.join(project_root, 'datasets')

#模型保存路徑
model_save_path = os.path.join(project_root, 'models')
os.makedirs(model_save_path, exist_ok=True)

#結果保存路徑
results_path = os.path.join(project_root, 'results')
os.makedirs(results_path, exist_ok=True)

print(f"數據路徑: {data_base_path}")
print(f"模型保存路徑: {model_save_path}")
print(f"結果保存路徑: {results_path}")

#改進的參數設定
img_width, img_height = 128, 128
batch_size = 16  #減小batch size提高泛化能力

#更強且多樣化的數據增強
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,          #增加旋轉範圍
    width_shift_range=0.2,      #增加位移範圍
    height_shift_range=0.2,
    shear_range=0.2,           #添加剪切變換
    zoom_range=0.2,            #增加縮放範圍
    horizontal_flip=True,
    vertical_flip=False,        #對音頻頻譜圖，垂直翻轉可能不合適
    fill_mode='nearest',
    brightness_range=[0.8, 1.2], #亮度變化
    channel_shift_range=0.1     #顏色通道變化
)

#訓練組
train_path = os.path.join(data_base_path, 'train')
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

#驗證組 - 只做標準化
val_path = os.path.join(data_base_path, 'val')
validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

#測試組
test_path = os.path.join(data_base_path, 'test')
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

#改進的CNN模型 - 更輕量化，更多正規化
model = Sequential()

#Block 1 - 減少參數數量
model.add(Conv2D(16, (3,3), activation='relu', padding='same', 
                 input_shape=(img_width, img_height, 3),
                 kernel_regularizer=l2(0.001)))  #添加 L2 正規化
model.add(BatchNormalization())
model.add(Conv2D(16, (3,3), activation='relu', padding='same',
                 kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))  #增加 dropout

#Block 2
model.add(Conv2D(32, (3,3), activation='relu', padding='same',
                 kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), activation='relu', padding='same',
                 kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#Block 3
model.add(Conv2D(64, (3,3), activation='relu', padding='same',
                 kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation='relu', padding='same',
                 kernel_regularizer=l2(0.001)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

#Block 4 - 最後一層卷積
model.add(Conv2D(128, (3,3), activation='relu', padding='same',
                 kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

#使用Global Average Pooling而非Flatten，減少參數
model.add(GlobalAveragePooling2D())

#更簡單的全連接層
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(train_generator.num_classes, activation='softmax'))

#編譯模型 - 使用更保守的學習率
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0005),  # 稍微提高初始學習率
    metrics=['accuracy']
)

print(model.summary())

#檢查數據載入
print(f"\n數據檢查:")
print(f"訓練數據: {train_generator.samples} 張圖片, {len(train_generator.class_indices)} 個類別")
print(f"驗證數據: {validation_generator.samples} 張圖片")
print(f"測試數據: {test_generator.samples} 張圖片")
print(f"類別對應: {train_generator.class_indices}")

#生成時間戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f'improved_model_{timestamp}.keras'
model_filepath = os.path.join(model_save_path, model_filename)

#改進的回調函數
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=20,  #增加耐心值
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,  #更溫和的學習率降低
        patience=10,  #增加耐心
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        model_filepath,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

#訓練模型
print(f"\n開始訓練改進模型...")
print(f"模型將保存至: {model_filepath}")

#計算每個epoch步數
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

train_history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=80,  # 減少最大epochs
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

#評估模型
print(f"\n評估改進模型...")
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"測試組準確率: {test_acc:.4f}")
print(f"測試組損失: {test_loss:.4f}")

#保存最終模型
final_model_path = os.path.join(model_save_path, f'final_improved_model_{timestamp}.keras')
model.save(final_model_path)
print(f"最終改進模型已保存至: {final_model_path}")

#繪製學習曲線
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_history.history['accuracy'], label='Training Accuracy')
plt.plot(train_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Improved Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_history.history['loss'], label='Training Loss')
plt.plot(train_history.history['val_loss'], label='Validation Loss')
plt.title('Improved Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(train_history.history['lr'], label='Learning Rate')
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.yscale('log')
plt.legend()

plt.tight_layout()

#保存學習曲線圖
curve_filename = f'improved_learning_curves_{timestamp}.png'
curve_filepath = os.path.join(results_path, curve_filename)
plt.savefig(curve_filepath, dpi=300, bbox_inches='tight')
print(f"改進模型學習曲線已保存至: {curve_filepath}")
plt.show()

#保存訓練歷史
import json
history_filename = f'improved_training_history_{timestamp}.json'
history_filepath = os.path.join(results_path, history_filename)

history_dict = {
    'accuracy': [float(x) for x in train_history.history['accuracy']],
    'val_accuracy': [float(x) for x in train_history.history['val_accuracy']],
    'loss': [float(x) for x in train_history.history['loss']],
    'val_loss': [float(x) for x in train_history.history['val_loss']],
    'lr': [float(x) for x in train_history.history['lr']],
    'test_accuracy': float(test_acc),
    'test_loss': float(test_loss),
    'num_classes': len(train_generator.class_indices),
    'class_indices': train_generator.class_indices,
    'model_path': model_filepath,
    'timestamp': timestamp,
    'improvements': [
        "更輕量化模型架構",
        "增強數據增強策略", 
        "添加L2正規化",
        "使用GlobalAveragePooling2D",
        "調整batch size和學習率",
        "更溫和的學習率衰減"
    ]
}

with open(history_filepath, 'w', encoding='utf-8') as f:
    json.dump(history_dict, f, indent=2, ensure_ascii=False)
    
print(f"改進模型訓練歷史已保存至: {history_filepath}")
print(f"\n改進模型訓練完成！")