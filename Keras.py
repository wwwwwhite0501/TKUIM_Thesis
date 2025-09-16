import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import matplotlib.pyplot as plt

#載入訓練、驗證、測試資料
X_train = np.load(r'C:\Users\USER\OneDrive\桌面\專題\npy\vggish_features_train.npy')
y_train = np.load(r'C:\Users\USER\OneDrive\桌面\專題\npy\vggish_labels_train.npy')
X_val = np.load(r'C:\Users\USER\OneDrive\桌面\專題\npy\vggish_features_val.npy')
y_val = np.load(r'C:\Users\USER\OneDrive\桌面\專題\npy\vggish_labels_val.npy')
X_test = np.load(r'C:\Users\USER\OneDrive\桌面\專題\npy\vggish_features_test.npy')
y_test = np.load(r'C:\Users\USER\OneDrive\桌面\專題\npy\vggish_labels_test.npy')

print(X_train.shape, X_val.shape, X_test.shape)

#檢查各組資料的類別分布
print("訓練組：", np.unique(y_train, return_counts=True))
print("驗證組：", np.unique(y_val, return_counts=True))
print("測試組：", np.unique(y_test, return_counts=True))

#標籤編碼(用同一個encoder)
le = LabelEncoder()
le.fit(np.concatenate([y_train, y_val, y_test]))
y_train_enc = to_categorical(le.transform(y_train))
y_val_enc = to_categorical(le.transform(y_val))
y_test_enc = to_categorical(le.transform(y_test))

#建立模型
model = Sequential([
    Dense(256, activation='relu', input_shape=(128,)),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y_train_enc.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#驗證組監控訓練
train_history = model.fit(X_train, y_train_enc, epochs=100, batch_size=16, validation_data=(X_val, y_val_enc))

#測試組評估
test_loss, test_acc = model.evaluate(X_test, y_test_enc)
print(f"測試組準確率: {test_acc:.4f}")

#儲存模型
model.save(r"C:/Users/USER/OneDrive/桌面/專題/Model/vggish_000.keras")

#繪製Keras訓練/驗證準確率曲線
plt.plot(train_history.history['accuracy'], label='Training Accuracy')
plt.plot(train_history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(r'C:/Users/USER/OneDrive/桌面/專題/Model/learning_curve_000.png')
plt.show()