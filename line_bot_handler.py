# line_bot_handler.py  (ALL features + Flex sanitize + QuickReply + Range reports + TZ fallback)

# ===== 時區 fallback：優先 IANA Asia/Taipei；沒有 tzdata 時退回 UTC+8 =====
try:
    from zoneinfo import ZoneInfo
    try:
        TAIPEI = ZoneInfo("Asia/Taipei")
    except Exception:
        from datetime import timezone, timedelta
        TAIPEI = timezone(timedelta(hours=8))
except Exception:
    from datetime import timezone, timedelta
    TAIPEI = timezone(timedelta(hours=8))
# ==========================================================================

import re
import threading
import time
from datetime import datetime, timedelta, time as dtime

from fastapi import HTTPException
from linebot import WebhookHandler, LineBotApi
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    MessageAction, TemplateSendMessage, ButtonsTemplate,
    FlexSendMessage, QuickReply, QuickReplyButton
)

from db import (
    get_db_connection, get_user_by_line_id,
    create_user_if_not_exists, map_user_to_device,
    list_devices_by_user, get_today_logs_for_user,
    get_logs_for_user_in_range, aggregate_logs_by_type_for_user_range
)
from config import LINE_CHANNEL_ACCESS_TOKEN, LINE_CHANNEL_SECRET
from audio_processor import pause_device, is_paused  # per-device pause helpers

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ========== 長文分段 ==========
MAX_TEXT_CHARS = 1800
def _split_text(text: str, size: int = MAX_TEXT_CHARS):
    return [text[i:i+size] for i in range(0, len(text), size)]

def reply_long_text(event, full_text: str):
    chunks = _split_text(full_text.strip())
    if not chunks:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="（空內容）"))
        return
    first_batch = [TextSendMessage(text=c) for c in chunks[:5]]
    line_bot_api.reply_message(event.reply_token, first_batch)
    uid = event.source.user_id
    i = 5
    while i < len(chunks):
        batch = [TextSendMessage(text=c) for c in chunks[i:i+5]]
        line_bot_api.push_message(uid, batch)
        i += 5

# ========== 寶寶攻略內文 ==========
BABY_LIST_TEXT = """［寶寶必備清單］
採購時機建議在懷孕7至8個月時，先依預算列清單；生產前2個月再分批入手即可。

🚼 嬰兒車與配件
• 嬰兒背帶：可騰出雙手，選堅固、易清洗。
• 嬰兒推車：外出便利、轉換心情。
• 兒童汽車安全座椅：法律規定，注意使用年限（約5~7年），避免過期或來路不明二手。

🍼 哺乳用品
• 餵奶專用枕、綿羊油、冰袋、防溢乳墊、圍兜。
• 奶瓶（玻璃/不鏽鋼）、配方奶（與醫師討論）。
• 集乳器（手壓/電動）、母乳儲存袋、擠乳組、溫奶器。
• 奶瓶刷、奶瓶清潔劑、消毒器（蒸汽/紫外線/化學）。
• 奶粉攜帶盒：外出方便。

🛏 新生兒寢具
• 毯子（可穿式/包巾式）、嬰兒床與床墊（檢驗合格、易清洗）。

🍽 餐飲用品
• 兒童湯匙（塑膠/橡膠材質）、兒童專用碗、兒童餐椅（可拆式托盤）、防水快乾圍兜。

🧴 生活與清潔必備
• 嬰兒床/床中床（結構牢固、避免銳角）。
• 紙尿布（依體重選 NB~XXL）。
• 固齒器、奶嘴（建議 ≥2 個替換）、嬰兒指甲剪。
• 體溫計/耳溫槍（3個月前以體溫計為主）。
• 嬰兒濕紙巾（天然溫和配方）。

🛁 盥洗用品
• 澡盆（可折疊）、嬰兒清潔用品（無香精、無塑化劑）。
• 嬰兒牙刷（6個月後）、紗布巾（≥5條）、大尺寸浴巾。

🎲 早教或其他
• 玩具（安全材質：絨毛、搖鈴、音樂娃娃）。
• 童書（啟發語言與想像力），搖搖椅（有安全綁帶）。

參考：
- https://www.gbding.com/blog/posts/newborn-baby-essentials-shopping-list#section2
- https://helloyishi.com.tw/parenting/babys-first-year/baby-care/things-to-buy-for-your-newborn/
- https://www.runnyyolk.com/blog/posts/2024新手爸媽必看！嬰兒用品必備清單，跟著買就對了
"""

CARE_GUIDE_TEXT = """［寶寶照護攻略］
一、親餵與瓶餵協助
● 親餵協助：
1) 姿勢舒適：準備哺乳枕/靠枕。
2) 防溢吐奶：紗布巾/小毛巾備用。
3) 口腔清潔：喝奶後以拋棄式紗布+溫開水輕拭。
4) 放鬆舒緩：按摩頭皮與肩頸，幫助媽媽放鬆。
● 瓶餵協助：
1) 輕點嘴唇：寶寶張口後再放入奶嘴。
2) 奶瓶傾斜：奶水填滿奶嘴，避免吞空氣。
3) 暫停拍嗝：不強迫喝完。
4) 口腔清潔：同上。
★ 小提醒：孔洞流速以「1秒1滴」最合適新生兒。

二、拍嗝
1) 側坐姿：寶寶側身坐在大人腿上，一手托住下巴/頸部/肩膀。
2) 空掌輕拍：另一手微彎空掌，輕拍背部。

三、洗澡
1) 橄欖球側抱：枕靠同側手臂，拇指輕按耳朵。
2) 先洗臉：布巾擰乾輕拭。
3) 再洗頭：少量洗沐用品，沖淨後擦乾。
4) 最後身體：注意後頸、腋下、生殖器等皺褶處。
★ 注意：
• 室溫25–28℃；先放冷水再放熱水。
• 水溫37–38℃；若洗後通紅或起疹，代表太熱需調降。
• 避免太餓或太飽時洗澡。

四、換尿布
1) 洗屁屁：優先溫開水（外出可用濕紙巾）。
2) 雙腳舉高：一手抓住雙腳抬起。
3) 放尿布：另一手置入乾淨尿布到位。
4) 黏好與檢查：確認腰部鬆緊、整理腿側邊。

五、臍帶護理
1) 先消毒：75%酒精，沿臍根由內向外繞一圈。
2) 後乾燥：95%酒精，再繞一圈。
★ 原則：保持清潔乾燥；臍帶未脫落前穿寬鬆、尿布反摺避濕。若有分泌惡臭、紅腫等臍帶炎，請就醫。

參考：
- https://mammy.hpa.gov.tw/Home/NewsKBContent?id=2403&type=01
- https://www.cmuh.cmu.edu.tw/HealthEdus/Detail?no=5220
- https://www.fhs.gov.hk/tc_chi/health_professional/OMP_eNewsletter/enews_20111031.html

延伸：
- 國民健康署-母乳哺育手冊 https://reurl.cc/jDKvRm
- 社家署-ㄜ!我打嗝了 https://reurl.cc/mDK0M7
- 社家署-寶寶洗澡和清潔 https://reurl.cc/94mR3n
- 高需求寶寶照護 https://www.chick.com.tw/baike-detail/育兒資訊站/寶寶居家照顧/high_need_baby
"""

VACCINE_TEXT = """［寶寶疫苗接種大全］
常見公費/自費疫苗摘要（實際依各地衛生單位公告）：
• B型肝炎疫苗：出生滿1個月接種。
• 卡介苗（BCG）：多在滿月後接種。
• 五合一：白喉、破傷風、非細胞性百日咳、Hib、不活化小兒麻痺。
• 13價肺炎鏈球菌（PCV13）。
• 水痘疫苗。
• MMR（麻疹、腮腺炎、德國麻疹）。
• A型肝炎疫苗。
• 日本腦炎疫苗。

常見副作用：注射處紅腫痛或輕微發燒，多於數日內緩解；若持續高燒或精神差，請就醫評估。

參考：
- https://nestlebaby.hk/content/0-至2-歲接種疫苗時間表
- https://kids.heho.com.tw/archives/154606
- https://www.carloine.com.tw/Article/Detail/78265
"""

# ---------- Flex 安全器：遞迴移除 color 類屬性 ----------
def _sanitize_flex_colors(obj):
    """
    遞迴刪除所有 color / backgroundColor / borderColor 屬性，
    防止 LINE 後端以 'invalid property' 退件。
    """
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            if k in ("color", "backgroundColor", "borderColor"):
                obj.pop(k, None)
            else:
                _sanitize_flex_colors(obj[k])
    elif isinstance(obj, list):
        for it in obj:
            _sanitize_flex_colors(it)
    return obj

# ---------- 重新定義：沒有 header、只有 body 按鈕 ----------
def build_baby_guide_menu():
    return {
        "type": "bubble",
        "size": "mega",
        "body": {
            "type": "box",
            "layout": "vertical",
            "spacing": "md",
            "contents": [
                {"type": "text", "text": "新生寶寶必修課 · 目錄", "weight": "bold", "size": "lg"},
                {"type": "separator", "margin": "md"},
                {"type": "button", "style": "primary",
                 "action": {"type": "message", "label": "寶寶必備清單", "text": "攻略-必備清單"}},
                {"type": "button", "style": "primary",
                 "action": {"type": "message", "label": "照護攻略（抱法/換尿布）", "text": "攻略-照護"}},
                {"type": "button", "style": "primary",
                 "action": {"type": "message", "label": "疫苗接種指南", "text": "攻略-疫苗"}}
            ]
        }
    }

def send_baby_guide_menu(event):
    try:
        flex = build_baby_guide_menu()
        flex = _sanitize_flex_colors(flex)  # 發送前保險刪色碼
        line_bot_api.reply_message(
            event.reply_token,
            FlexSendMessage(alt_text="寶寶攻略大全", contents=flex)
        )
    except Exception as e:
        print("[baby_guide_menu] send error:", repr(e), flush=True)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="無法顯示寶寶攻略選單，請稍後再試。")
        )

def handle_line_message_sync(body, signature):
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        print(f"LINE handler error: {e}")
        raise HTTPException(status_code=500, detail="LINE handler error")

# ========== 暫停解析 ==========
PAUSE_DEFAULT_SECONDS = 60
PAUSE_MAX_SECONDS = 4 * 60 * 60

def parse_pause_command(msg: str):
    """
    支援：
      - 暫停
      - 暫停 120
      - 暫停 B001
      - 暫停 B001 120
    """
    m = re.match(r"^暫停(?:偵測)?(?:\s+([A-Za-z0-9_\-]+))?(?:\s+(\d+))?$", msg.strip())
    if not m:
        return (None, None)
    serial = m.group(1)
    sec = m.group(2)
    seconds = int(sec) if sec is not None else None
    return (serial, seconds)

# ========== 區間工具 ==========
def _today_range(now: datetime | None = None):
    now = now.astimezone(TAIPEI) if now else datetime.now(TAIPEI)
    start = datetime.combine(now.date(), dtime.min, tzinfo=TAIPEI)
    end = start + timedelta(days=1)
    return start, end

def _this_week_range(now: datetime | None = None):
    now = now.astimezone(TAIPEI) if now else datetime.now(TAIPEI)
    this_monday = (now.date() - timedelta(days=now.weekday()))
    start = datetime.combine(this_monday, dtime.min, tzinfo=TAIPEI)
    end = start + timedelta(days=7)
    return start, end

def _this_month_range(now: datetime | None = None):
    now = now.astimezone(TAIPEI) if now else datetime.now(TAIPEI)
    start = datetime(now.year, now.month, 1, tzinfo=TAIPEI)
    if now.month == 12:
        end = datetime(now.year + 1, 1, 1, tzinfo=TAIPEI)
    else:
        end = datetime(now.year, now.month + 1, 1, tzinfo=TAIPEI)
    return start, end

def _last_month_range(now: datetime | None = None):
    now = now.astimezone(TAIPEI) if now else datetime.now(TAIPEI)
    if now.month == 1:
        start = datetime(now.year - 1, 12, 1, tzinfo=TAIPEI)
        end = datetime(now.year, 1, 1, tzinfo=TAIPEI)
    else:
        start = datetime(now.year, now.month - 1, 1, tzinfo=TAIPEI)
        end = datetime(now.year, now.month, 1, tzinfo=TAIPEI)
    return start, end

def _fmt_ts_local(ts):
    if not ts:
        return "--:--"
    if getattr(ts, "tzinfo", None) is None:
        try:
            ts = ts.replace(tzinfo=TAIPEI)
        except Exception:
            pass
    try:
        return ts.astimezone(TAIPEI).strftime("%m/%d %H:%M")
    except Exception:
        return ts.strftime("%m/%d %H:%M")

def _format_range_report(line_user_id: str, cursor, title: str, start_dt: datetime, end_dt: datetime) -> str:
    rows = get_logs_for_user_in_range(cursor, line_user_id, start_dt, end_dt)
    aggs = aggregate_logs_by_type_for_user_range(cursor, line_user_id, start_dt, end_dt)

    total = len(rows)
    head = f"📊 {title}（{start_dt.strftime('%Y-%m-%d')} ~ {(end_dt - timedelta(seconds=1)).strftime('%Y-%m-%d')}）\n總次數：{total}\n"

    if aggs:
        agg_lines = []
        for r in aggs:
            kind = r["kind"] if isinstance(r, dict) else r[0]
            cnt  = r["cnt"]  if isinstance(r, dict) else r[1]
            agg_lines.append(f"- {kind}：{cnt}")
        head += "\n📈 類別統計：\n" + "\n".join(agg_lines) + "\n"
    else:
        head += "\n📈 類別統計：\n（無資料）\n"

    if rows:
        detail_lines = []
        for i, r in enumerate(rows[:50]):
            cry = (r["cry_type"] if isinstance(r, dict) else r[0]) or ""
            reason = (r["reason"] if isinstance(r, dict) else r[1]) or ""
            ts = (r["ts"] if isinstance(r, dict) else r[2])
            label = cry or reason or "未知"
            detail_lines.append(f"- {_fmt_ts_local(ts)}：{label}")
        if len(rows) > 50:
            detail_lines.append(f"... 其餘 {len(rows) - 50} 筆省略")
        return head + "\n📝 明細：\n" + "\n".join(detail_lines)
    else:
        return head + "\n📝 明細：\n（無資料）"

def _resume_after_delay(user_id: str, seconds: int = 60):
    time.sleep(max(1, seconds))
    try:
        line_bot_api.push_message(user_id, TextSendMessage(text="✅恢復偵測"))
    except Exception as e:
        print(f"[resume_after_delay] push error:", repr(e))

# ===================== 事件處理 =====================
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    msg = event.message.text.strip()

    # 1) 暫停偵測（支援：暫停 / 暫停 120 / 暫停 B001 / 暫停 B001 120）
    if msg.startswith("暫停"):
        try:
            target_serial, seconds = parse_pause_command(msg)
            db = get_db_connection()
            cur = db.cursor(dictionary=True)
            u = get_user_by_line_id(cur, user_id)
            if not u:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="您尚未加入任何裝置"))
                db.close()
                return
            owned_serials = list_devices_by_user(cur, u["id"])
            db.close()

            owned_serials = [r if isinstance(r, str) else (r.get("device_id") if isinstance(r, dict) else str(r)) for r in (owned_serials or [])]
            owned_serials = [s for s in owned_serials if s]

            if seconds is None:
                seconds = PAUSE_DEFAULT_SECONDS
            seconds = max(1, min(PAUSE_MAX_SECONDS, int(seconds)))

            if target_serial:
                if target_serial not in owned_serials:
                    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"您尚未綁定裝置 {target_serial}，無法暫停"))
                    return
                to_pause = [target_serial]
            else:
                to_pause = owned_serials

            if any(is_paused(s) for s in to_pause):
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="系統已在暫停中⏳"))
                return

            for s in to_pause:
                pause_device(s, seconds=seconds)

            if len(to_pause) == 1:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"⏸ 已暫停 {to_pause[0]} {seconds} 秒"))
            else:
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"⏸ 已暫停您名下 {len(to_pause)} 台裝置 {seconds} 秒"))
            threading.Thread(target=_resume_after_delay, args=(user_id, seconds), daemon=True).start()
        except Exception as e:
            print("[pause] error:", repr(e), flush=True)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"暫停指令發生錯誤：{e}"))
        return

    # 後續查 DB
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    # 2) 加入裝置（加入B001）
    if msg.startswith("加入"):
        device_code = msg.replace("加入", "").strip()
        cursor.execute("SELECT id FROM devices WHERE device_id = %s", (device_code,))
        device = cursor.fetchone()
        if not device:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="裝置不存在"))
            db.close()
            return
        try:
            profile = line_bot_api.get_profile(user_id)
            display_name = profile.display_name
        except Exception:
            display_name = "用戶"

        user = create_user_if_not_exists(db, cursor, user_id, display_name)
        added = map_user_to_device(db, cursor, device["id"], user["id"])
        if added:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"{display_name} 已加入裝置 {device_code}"))
        else:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"你已加入裝置 {device_code}"))
        db.close()
        return

    # 3) 顯示序號登入成員 / 裝置列表
    if msg in ["顯示序號登入成員", "顯示已加入裝置", "裝置列表"]:
        cursor.execute("SELECT id FROM users WHERE line_user_id = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="您尚未加入任何裝置。"))
            db.close()
            return

        cursor.execute("""
            SELECT d.device_id FROM devices d
            JOIN device_user_map m ON d.id = m.device_id
            WHERE m.user_id = %s
        """, (user["id"],))
        devices = cursor.fetchall()
        if not devices:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="您尚未加入任何裝置。"))
            db.close()
            return

        result_msg = "你已加入的裝置：\n"
        for device in devices:
            result_msg += f"\n序號：{device['device_id']}\n"
            cursor.execute("""
                SELECT u.display_name
                FROM users u
                JOIN device_user_map m ON u.id = m.user_id
                WHERE m.device_id = (SELECT id FROM devices WHERE device_id = %s)
            """, (device["device_id"],))
            members = cursor.fetchall()
            names = [m["display_name"] or "匿名用戶" for m in members]
            result_msg += f"成員：{', '.join(names)}\n"

        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=result_msg))
        db.close()
        return

    # 4) 哭聲紀錄選單（QuickReply）
    if msg in ["3", "3."] or ("哭聲紀錄" in msg):
        qr = QuickReply(items=[
            QuickReplyButton(action=MessageAction(label="今日",   text="紀錄-今日")),
            QuickReplyButton(action=MessageAction(label="本周",   text="紀錄-本周")),
            QuickReplyButton(action=MessageAction(label="本月",   text="紀錄-本月")),
            QuickReplyButton(action=MessageAction(label="上個月", text="紀錄-上個月")),
        ])
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="請選擇要查詢的區間：", quick_reply=qr)
        )
        db.close()
        return

    # —— 各區間查詢 —— #
    if msg in ["紀錄-今日", "紀錄-本周", "紀錄-本月", "紀錄-上個月"]:
        if msg == "紀錄-今日":
            start_dt, end_dt = _today_range()
            title = "今日哭聲紀錄"
        elif msg == "紀錄-本周":
            start_dt, end_dt = _this_week_range()
            title = "本周哭聲紀錄"
        elif msg == "紀錄-本月":
            start_dt, end_dt = _this_month_range()
            title = "本月哭聲紀錄"
        else:
            start_dt, end_dt = _last_month_range()
            title = "上個月哭聲紀錄"

        try:
            report = _format_range_report(user_id, cursor, title, start_dt, end_dt)
            reply_long_text(event, report)
        except Exception as e:
            print("[logs-range] error:", repr(e), flush=True)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"查詢失敗：{e}"))
        db.close()
        return

    # 5) 寶寶攻略大全（Flex 選單 + 長文）
    if msg in ["4", "4."] or ("寶寶攻略" in msg):
        send_baby_guide_menu(event)
        db.close()
        return

    if msg in ["攻略-必備清單", "寶寶必備清單"]:
        reply_long_text(event, BABY_LIST_TEXT); db.close(); return
    if msg in ["攻略-照護", "寶寶照護攻略", "照護攻略"]:
        reply_long_text(event, CARE_GUIDE_TEXT); db.close(); return
    if msg in ["攻略-疫苗", "寶寶疫苗接種指南", "疫苗接種指南"]:
        reply_long_text(event, VACCINE_TEXT); db.close(); return

    # 沒有 fallback 提示（依你的要求）
    db.close()
