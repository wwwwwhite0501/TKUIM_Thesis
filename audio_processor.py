# -*- coding: utf-8 -*-
from __future__ import annotations
import asyncio
import time
import cv2
import threading
import numpy as np
from typing import Dict, List

from model import preprocess_audio, predict_model  
from db import get_db_connection, get_device_id, log_cry
from notifier import notify_line_users
from config import CLASS_NAMES 
from music_player import music 

# ====== 1. 鏡頭監看控制 ======
_video_end_time = 0.0  

def camera_worker():
    global _video_end_time
    cap = None
    window_name = "Baby Monitor - Live View"
    while True:
        now = time.time()
        if now < _video_end_time:
            if cap is None:
                cap = cv2.VideoCapture(0) # 開啟預設鏡頭
                print("📸 偵測到活動，啟動鏡頭監看視窗", flush=True)
            ret, frame = cap.read()
            if ret:
                cv2.imshow(window_name, frame)
                cv2.waitKey(1) # 刷新視窗必備
        else:
            if cap is not None:
                cap.release()
                cap = None
                cv2.destroyWindow(window_name)
                print("🛑 10 秒未偵測到活動，關閉監看視窗", flush=True)
        time.sleep(0.03)

threading.Thread(target=camera_worker, daemon=True).start()

# ====== 2. 裝置控制與緩衝區 ======
audio_buffers: Dict[str, List[int]] = {}
_paused_until: Dict[str, float] = {}

def pause_device(serial: str, seconds: int = 60):
    until = time.time() + max(1, int(seconds))
    _paused_until[serial] = until
    print(f"⏸ 裝置 {serial} 暫停至 {time.strftime('%H:%M:%S', time.localtime(until))}", flush=True)

def is_paused(serial: str) -> bool:
    until = _paused_until.get(serial, 0)
    return time.time() < until

# ====== 3. 處理參數 ======
WINDOW_SIZE = 32000 
STEP_SIZE = 16000   
SMOOTH_K = 5        
CONF_THR = 0.6      # 通知門檻
MIN_RMS = 0.01      

async def process_audio_loop():
    recent_preds: Dict[str, List[np.ndarray]] = {}
    loop = asyncio.get_event_loop()
    global _video_end_time 

    while True:
        for serial in list(audio_buffers.keys()):
            if is_paused(serial): continue
            buffer = audio_buffers[serial]

            while len(buffer) >= WINDOW_SIZE:
                window_audio = buffer[:WINDOW_SIZE]
                wa = np.asarray(window_audio, dtype=np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(wa ** 2)) + 1e-9)

                if rms < MIN_RMS:
                    buffer = buffer[STEP_SIZE:]
                    audio_buffers[serial] = buffer
                    continue

                # --------- 推論 ---------
                input_data = preprocess_audio(window_audio)
                probs = await loop.run_in_executor(None, predict_model, input_data)

                # 🔎 格式化輸出機率
                msg_probs = " ".join(f"{CLASS_NAMES[i]}:{probs[i]:.3f}" for i in range(len(CLASS_NAMES)))
                print(f"🔎 probs → {msg_probs}", flush=True)

                # 🧪 格式化輸出視窗結果
                idx = int(np.argmax(probs))
                label = CLASS_NAMES[idx]
                conf = float(probs[idx])
                print(f"🧪 {serial} Window：{label} (信心 {conf:.2f})", flush=True)

                # --------- 平滑化處理 ---------
                recent_preds.setdefault(serial, []).append(probs)
                if len(recent_preds[serial]) > SMOOTH_K:
                    recent_preds[serial].pop(0)
                
                avg_pred = np.mean(recent_preds[serial], axis=0)
                avg_idx = int(np.argmax(avg_pred))
                avg_label = CLASS_NAMES[avg_idx]
                avg_conf = float(avg_pred[avg_idx])

                # 🎯 格式化輸出平均結果
                print(f"🎯 {serial} 平均：{avg_label} (信心 {avg_conf:.2f})", flush=True)

                # --------- 核心邏輯：辨識成功同時觸發 鏡頭 + 音樂 + 通知 ---------
                if avg_label != "雜訊" and avg_conf >= CONF_THR:
                    # 1. 觸發鏡頭 (更新結束時間)
                    _video_end_time = time.time() + 10 
                    print(f"⏰ {serial} 觸發辨識事件，監看視窗重設為 10 秒", flush=True)

                    # 2. 觸發音樂
                    if not music.is_playing():
                        print("🎵 觸發 60 秒安撫音樂", flush=True)
                        await loop.run_in_executor(None, music.play_random, 60)
                    
                    # 3. 記錄與發送 LINE 通知
                    try:
                        db = get_db_connection()
                        cursor = db.cursor()
                        device_id = get_device_id(cursor, serial)
                        if device_id:
                            log_cry(db, device_id, avg_label, avg_conf)
                            db.close()
                            # 執行通知
                            await loop.run_in_executor(None, notify_line_users, serial, 
                                                     f"寶寶哭聲辨識結果：{avg_label} (信心 {avg_conf:.2f})")
                    except Exception as e:
                        print(f"❗ 記錄/通知失敗：{e}", flush=True)
                else:
                    # ⚠️ 信心值過低輸出
                    print(f"⚠️ {serial} 信心值過低（{avg_conf:.2f}），不通知", flush=True)

                # --------- 滑動視窗 ---------
                buffer = buffer[STEP_SIZE:]
                audio_buffers[serial] = buffer

        await asyncio.sleep(0.01)