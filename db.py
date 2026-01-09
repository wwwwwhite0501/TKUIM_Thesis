# db.py
import mysql.connector
from config import MYSQL_CONFIG
from datetime import datetime

def get_db_connection():
    return mysql.connector.connect(**MYSQL_CONFIG)

def get_device_id(cursor, device_serial):
    cursor.execute("SELECT id FROM devices WHERE device_id = %s", (device_serial,))
    result = cursor.fetchone()
    if result:
        return result[0] if not isinstance(result, dict) else result.get("id")
    return None

def log_cry(db, device_id, cry_type, confidence):
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO cry_logs (device_id, cry_type, confidence, audio_path, timestamp, notified)
        VALUES (%s, %s, %s, '', NOW(), FALSE)
    """, (device_id, cry_type, float(confidence)))
    db.commit()
    cursor.close()

def get_users_by_device(cursor, device_id):
    cursor.execute("""
        SELECT users.line_user_id
        FROM users
        JOIN device_user_map ON users.id = device_user_map.user_id
        WHERE device_user_map.device_id = %s
    """, (device_id,))
    rows = cursor.fetchall()
    out = []
    for r in rows:
        out.append(r[0] if not isinstance(r, dict) else r.get("line_user_id"))
    return out

# ---------- LINE helpers ----------
def get_user_by_line_id(cursor, line_user_id: str):
    cursor.execute("SELECT * FROM users WHERE line_user_id = %s", (line_user_id,))
    return cursor.fetchone()

def create_user_if_not_exists(db, cursor, line_user_id: str, display_name: str):
    u = get_user_by_line_id(cursor, line_user_id)
    if u:
        return u
    cursor.execute(
        "INSERT INTO users (line_user_id, display_name) VALUES (%s, %s)",
        (line_user_id, display_name)
    )
    db.commit()
    cursor.execute("SELECT * FROM users WHERE line_user_id = %s", (line_user_id,))
    return cursor.fetchone()

def map_user_to_device(db, cursor, device_id: int, user_id: int) -> bool:
    cursor.execute(
        "SELECT 1 FROM device_user_map WHERE device_id = %s AND user_id = %s",
        (device_id, user_id)
    )
    if cursor.fetchone():
        return False
    cursor.execute(
        "INSERT INTO device_user_map (device_id, user_id) VALUES (%s, %s)",
        (device_id, user_id)
    )
    db.commit()
    return True

def list_devices_by_user(cursor, user_id: int):
    cursor.execute("""
        SELECT d.device_id
        FROM devices d
        JOIN device_user_map m ON d.id = m.device_id
        WHERE m.user_id = %s
    """, (user_id,))
    rows = cursor.fetchall()
    serials = []
    for r in rows:
        serials.append(r.get("device_id") if isinstance(r, dict) else r[0])
    return [s for s in serials if s]

def get_today_logs_for_user(cursor, line_user_id: str):
    cursor.execute("SELECT id FROM users WHERE line_user_id=%s", (line_user_id,))
    u = cursor.fetchone()
    if not u:
        return []
    user_id = u["id"] if isinstance(u, dict) else u[0]

    cursor.execute("""
        SELECT cl.cry_type, cl.reason,
               COALESCE(cl.created_at, cl.timestamp) AS created_at
        FROM cry_logs cl
        JOIN device_user_map m ON cl.device_id = m.device_id
        WHERE m.user_id = %s
          AND DATE(COALESCE(cl.created_at, cl.timestamp)) = CURDATE()
        ORDER BY COALESCE(cl.created_at, cl.timestamp) ASC
    """, (user_id,))
    return cursor.fetchall()

# ---------- 新增：區間查詢 ----------
def get_logs_for_user_in_range(cursor, line_user_id: str, start_dt: datetime, end_dt: datetime):
    """
    取得使用者名下裝置在 [start_dt, end_dt) 的 cry_logs 清單。
    回傳欄位：cry_type, reason, ts(=COALESCE(created_at, timestamp)), device_id。
    """
    cursor.execute("SELECT id FROM users WHERE line_user_id=%s", (line_user_id,))
    u = cursor.fetchone()
    if not u:
        return []
    user_id = u["id"] if isinstance(u, dict) else u[0]

    cursor.execute("""
        SELECT cl.cry_type, cl.reason,
               COALESCE(cl.created_at, cl.timestamp) AS ts,
               cl.device_id
        FROM cry_logs cl
        JOIN device_user_map m ON cl.device_id = m.device_id
        WHERE m.user_id = %s
          AND COALESCE(cl.created_at, cl.timestamp) >= %s
          AND COALESCE(cl.created_at, cl.timestamp) < %s
        ORDER BY ts ASC
    """, (user_id, start_dt, end_dt))
    return cursor.fetchall()

def aggregate_logs_by_type_for_user_range(cursor, line_user_id: str, start_dt: datetime, end_dt: datetime):
    """
    回傳各類型統計：[(kind, cnt), ...]，kind 來自 cry_type 或 reason（兩者其一），無則 '未知'
    """
    cursor.execute("SELECT id FROM users WHERE line_user_id=%s", (line_user_id,))
    u = cursor.fetchone()
    if not u:
        return []
    user_id = u["id"] if isinstance(u, dict) else u[0]

    cursor.execute("""
        SELECT COALESCE(cl.cry_type, cl.reason, '未知') AS kind, COUNT(*) AS cnt
        FROM cry_logs cl
        JOIN device_user_map m ON cl.device_id = m.device_id
        WHERE m.user_id = %s
          AND COALESCE(cl.created_at, cl.timestamp) >= %s
          AND COALESCE(cl.created_at, cl.timestamp) < %s
        GROUP BY kind
        ORDER BY cnt DESC
    """, (user_id, start_dt, end_dt))
    return cursor.fetchall()
