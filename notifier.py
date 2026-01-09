from linebot import LineBotApi
from linebot.models import TextSendMessage
from config import LINE_CHANNEL_ACCESS_TOKEN
from db import get_db_connection, get_device_id, get_users_by_device

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)

def notify_line_users(device_serial, message):
    db = get_db_connection()
    cursor = db.cursor()
    device_id = get_device_id(cursor, device_serial)
    if device_id is None:
        print(f"[錯誤] 裝置 {device_serial} 不存在")
        db.close()
        return

    user_ids = get_users_by_device(cursor, device_id)
    db.close()

    for user_id in user_ids:
        try:
            line_bot_api.push_message(user_id, TextSendMessage(text=message))
            print(f"✅ 已通知 {user_id}")
        except Exception as e:
            print(f"[錯誤] LINE 通知失敗：{e}")
