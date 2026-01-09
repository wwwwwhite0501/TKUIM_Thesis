import sys
import io
import asyncio

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse

from line_bot_handler import handle_line_message_sync
from audio_processor import audio_buffers, process_audio_loop

# 強制 stdout utf-8（避免 emoji 亂碼）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

app = FastAPI()

@app.post("/upload_data/{serial}")
async def upload_data(serial: str, request: Request):
    if serial not in audio_buffers:
        audio_buffers[serial] = []

    raw = await request.body()
    # 轉 int16 little-endian
    samples = [int.from_bytes(raw[i:i+2], 'little', signed=True)
               for i in range(0, len(raw), 2)]
    audio_buffers[serial].extend(samples)

    print(f"📥 裝置 {serial} 收到 {len(samples)} 筆樣本，累積至 {len(audio_buffers[serial])}", flush=True)
    return JSONResponse(content={"status": "received", "length": len(samples), "buffer_length": len(audio_buffers[serial])})

@app.post("/webhook")
async def line_webhook(request: Request, x_line_signature: str = Header(None)):
    if not x_line_signature:
        raise HTTPException(status_code=400, detail="Missing X-Line-Signature")
    body = await request.body()
    body_str = body.decode()
    loop = asyncio.get_event_loop()
    # 使用執行緒避免阻塞事件迴圈
    await loop.run_in_executor(None, handle_line_message_sync, body_str, x_line_signature)
    return "OK"

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_audio_loop())
