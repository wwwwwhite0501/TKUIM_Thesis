import torch
import torch.nn as nn
import numpy as np
import librosa
# 這裡修正為正確的類別名稱
from Transformer import BabyCryTransformer

# 設定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "transformer_model_001.pth"

# 根據你的 train_transformer_full.py CONFIG 初始化模型
# input_dim 必須是 128 (對應 n_mels)
model = BabyCryTransformer(
    input_dim=128, 
    num_classes=7, 
    nhead=4, 
    num_layers=3, 
    dim_feedforward=256
).to(DEVICE)

# 載入權重 (因為你是存成 dict，需要抓 model_state)
try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"✅ 成功載入 Transformer 模型: {MODEL_PATH}")
except Exception as e:
    print(f"❌ 載入失敗: {e}")

def preprocess_audio(window_audio):
    # 轉為浮點數
    y = np.array(window_audio).astype(np.float32) / 32768.0
    # 提取 MFCC 或 MelSpectrogram (訓練是用 128 階 Mel)
    # 這裡建議改用 librosa.feature.melspectrogram 並轉為 dB
    S = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=1024, hop_length=256, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def predict_model(mel_data):
    # 模型預期 (B, T, C)，目前 mel_data 是 (128, T) -> 轉置為 (T, 128)
    t_data = torch.from_numpy(mel_data).float().to(DEVICE)
    t_data = t_data.permute(1, 0).unsqueeze(0) # (1, T, 128)
    
    with torch.no_grad():
        output = model(t_data)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    return probs