import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf  # 支援 mp3, flac 等音檔格式

# ======================== 基本參數 ========================
# 音檔資料夾
audio_directory = r'C:/Users/USER/OneDrive/桌面/專題/BabyCryDataset3371/xxxxx'
# 圖片儲存資料夾
output_image_dir = os.path.join(r'C:/Users/USER/OneDrive/桌面/專題/BabyCryDataset頻譜圖', 'xxxxx')
# 支援的音檔格式
supported_extensions = ('.wav', '.mp3', '.flac', '.ogg')

# ======================== Mel 頻譜圖產生函數 ========================
def convert_audio_to_spectrogram_image(audio_file_path, output_image_path, sr=22050, n_fft=2048, hop_length=256, n_mels=128):
    if not os.path.exists(audio_file_path):
        print(f"錯誤：找不到音檔 '{audio_file_path}'")
        return False
    try:
        y, sr = librosa.load(audio_file_path, sr=sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.axis('off')  # 隱藏座標軸
        plt.tight_layout()
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
        print(f"轉換完成：{output_image_path}")
        return True
    except Exception as e:
        print(f"轉換失敗：{e}")
        return False

# ======================== 主程式邏輯 ========================
if __name__ == "__main__":
    print(f"\n--- 開始掃描音檔 ---")
    print(f"目錄：{audio_directory}")

    if not os.path.exists(audio_directory):
        print("錯誤：找不到資料夾，請確認路徑是否正確")
        exit()

    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
        print(f"已建立圖檔資料夾：{output_image_dir}")

    files = os.listdir(audio_directory)
    audio_files = [f for f in files if f.lower().endswith(supported_extensions) and os.path.isfile(os.path.join(audio_directory, f))]

    if not audio_files:
        print("未偵測到音檔，開始產生測試音檔...")
        test_file = os.path.join(audio_directory, 'test_audio.wav')
        duration = 5
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(test_file, tone, sr)
        audio_files = ['test_audio.wav']

    print(f"偵測到音檔：{audio_files}")

    for filename in audio_files:
        input_path = os.path.join(audio_directory, filename)
        output_path = os.path.join(output_image_dir, os.path.splitext(filename)[0] + '.png')
        convert_audio_to_spectrogram_image(input_path, output_path)

    print(f"\n所有音檔轉換完成，請查看資料夾：{output_image_dir}")