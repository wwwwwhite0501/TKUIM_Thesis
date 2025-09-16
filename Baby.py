import serial
import time
import requests
import wave
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from linebot.v3.messaging import MessagingApi, Configuration, ApiClient
from linebot.v3.messaging.models import PushMessageRequest, TextMessage

# ========== LINE 設定 ==========
LINE_CHANNEL_ACCESS_TOKEN = 'p6sj7qQEZqNIGjg6kyDLqAP/DkznyQYwXqCQZTREuu9M8zC8pLlbj88Y++fIfyKI4zSnfP/nRr5/Nk43R0c15gi5pvAmnJRy8/LtKdvP4iMGGkyPSyxqzwztXPAN3ROKEsCsk5weWXgVOCme/jhTxQdB04t89/1O/w1cDnyilFU='
LINE_USER_ID = '2007388217'

# ========== 串口與檔案設定 ==========
SERIAL_PORT = 'COM8'  # 根據你的 Arduino 調整
BAUD_RATE = 115200
AUDIO_FILE = 'baby.wav'
IMAGE_FILE = 'mel_spectrogram.png'

# ========== 錄音並儲存為 .wav ==========
def write_wav(audio_bytes, filename, sample_rate=16000):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)

def record_audio(timeout=5):
    print("開始錄音...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    audio = b''
    start_time = time.time()

    while True:
        if ser.in_waiting:
            audio += ser.read(ser.in_waiting)
            start_time = time.time()
        if time.time() - start_time > timeout:
            break

    ser.close()
    write_wav(audio, AUDIO_FILE)
    print(f"音檔儲存完成：{AUDIO_FILE}（長度約 {len(audio)/32000:.2f} 秒）")

# ========== 音檔轉換為 Mel 頻譜圖 ==========
def generate_mel_spectrogram(audio_path, output_image):
    print("轉換 Mel 頻譜圖...")
    np.complex = complex  # 修正 librosa 相容問題
    y, sr = librosa.load(audio_path, sr=None)
    if len(y) == 0:
        print("音訊為空，無法產生頻譜圖")
        return False

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    if os.path.exists(output_image):
        print(f"頻譜圖已儲存：{output_image}")
        return True
    else:
        print("頻譜圖儲存失敗")
        return False

# ========== 上傳音檔 ==========
def upload_file(filepath):
    print("上傳音檔至 transfer.sh...")
    try:
        with open(filepath, 'rb') as f:
            filename = os.path.basename(filepath)
            response = requests.put(f"https://transfer.sh/{filename}", data=f)
        if response.status_code == 200:
            url = response.text.strip()
            print("上傳成功：", url)
            return url
        else:
            print("上傳失敗，狀態碼：", response.status_code)
            return None
    except Exception as e:
        print("上傳時錯誤：", str(e))
        return None

# ========== 發送 LINE Bot 訊息 ==========
def send_line_message(message_text):
    print("傳送 LINE Bot 訊息...")
    try:
        config = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
        with ApiClient(config) as api_client:
            line_bot_api = MessagingApi(api_client)
            message = TextMessage(text=message_text)
            request = PushMessageRequest(to=LINE_USER_ID, messages=[message])
            line_bot_api.push_message(request)
            print("LINE 訊息已發送")
    except Exception as e:
        print("發送 LINE 訊息失敗：", str(e))

# ========== 主程式 ==========
if __name__ == '__main__':
    record_audio()
    success = generate_mel_spectrogram(AUDIO_FILE, IMAGE_FILE)
    file_url = upload_file(AUDIO_FILE)
    
    if file_url:
        send_line_message(f"偵測到聲音\n音檔連結：{file_url}")
    else:
        send_line_message("錄音上傳失敗")

    if not success:
        print("頻譜圖轉換未完成")