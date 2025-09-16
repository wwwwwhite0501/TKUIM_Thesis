import pygame
import random
import os
import serial

MUSIC_DIR = r"C:\Users\USER\OneDrive\桌面\專題\Music"
SERIAL_PORT = 'COM8' #Arduino序列埠
BAUDRATE = 115200 #鮑率

def play_random_music():
    #找出所有mp3檔
    music_files = [f for f in os.listdir(MUSIC_DIR) if f.endswith(".mp3")]
    if not music_files:
        print("⚠️沒有找到任何音樂檔")
        return
    #隨機選一首載入播放
    file = random.choice(music_files)
    path = os.path.join(MUSIC_DIR, file)
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    print(f"正在播放：{file}")

def stop_music():
    pygame.mixer.music.stop()
    print("音樂已停止")
    #等待音樂播放結束
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

if __name__ == "__main__":
    pygame.mixer.init()
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    print("等待Arduino指令...")
    while True:
        if ser.in_waiting:
            line = ser.readline().decode().strip()
            print(f"收到訊號：{line}")
            if line == "PLAY":
                play_random_music()
                # 播放 1 分鐘後自動停止
                time.sleep(60)
                stop_music()
            elif line == "STOP":
                stop_music()