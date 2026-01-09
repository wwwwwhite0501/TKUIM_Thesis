# -*- coding: utf-8 -*-
# music_player.py
# 單機版安撫音樂控制（使用 pygame.mixer）

import os
import random
import threading

import pygame

# 改成你放 mp3 的資料夾
MUSIC_DIR = r"C:\Users\hanha\Desktop\專題\0106\baby_cry_project\Music"

class MusicController:
    def __init__(self):
        # 初始化 mixer（若已經初始化會丟錯，包 try 避免重複）
        try:
            pygame.mixer.init()
        except Exception:
            pass
        self._lock = threading.RLock()
        self._timer: threading.Timer | None = None

    def _cancel_timer(self):
        if self._timer:
            try:
                self._timer.cancel()
            except Exception:
                pass
        self._timer = None

    def is_playing(self) -> bool:
        """目前是否有音樂在播放"""
        try:
            return pygame.mixer.music.get_busy()
        except Exception:
            return False

    def stop(self):
        """立刻停止播放，並清掉計時器"""
        with self._lock:
            self._cancel_timer()
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            print("⏹ 音樂已停止", flush=True)

    def _stop_after(self, seconds: int):
        """背景計時器到時候呼叫，用來自動停止"""
        def _do_stop():
            self.stop()
        self._cancel_timer()
        self._timer = threading.Timer(max(0, int(seconds)), _do_stop)
        self._timer.daemon = True
        self._timer.start()

    def play_random(self, seconds: int = 60, volume: float = 0.8):
        """
        從 MUSIC_DIR 隨機挑一首 mp3 播放；seconds 秒後自動停止。
        若目前正在播放則直接略過，不重啟。
        """
        with self._lock:
            if self.is_playing():
                print("🎵 已在播放中，略過重新播放", flush=True)
                return

            if not os.path.isdir(MUSIC_DIR):
                print(f"⚠️ 找不到音樂資料夾：{MUSIC_DIR}", flush=True)
                return

            files = [f for f in os.listdir(MUSIC_DIR) if f.lower().endswith(".mp3")]
            if not files:
                print("⚠️ 資料夾內沒有 .mp3 檔", flush=True)
                return

            pick = random.choice(files)
            path = os.path.join(MUSIC_DIR, pick)

            try:
                pygame.mixer.music.load(path)
                pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))
                pygame.mixer.music.play()
                print(f"▶️ 正在播放：{pick}（將在 {seconds} 秒後自動停止）", flush=True)
            except Exception as e:
                print(f"❗ 播放失敗：{e}", flush=True)
                return

            # 設定自動停止倒數
            self._stop_after(seconds)

# 匯出單例供其他模組使用
music = MusicController()
