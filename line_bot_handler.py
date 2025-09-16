import asyncio
from fastapi import HTTPException
from linebot import WebhookHandler, LineBotApi
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from db import (
    get_db_connection, get_device_id, get_user_by_line_id,
    create_user_if_not_exists, map_user_to_device, list_devices_by_user,
    get_today_logs_for_user
)
from config import LINE_CHANNEL_ACCESS_TOKEN, LINE_CHANNEL_SECRET
from audio_processor import pause_device

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

def handle_line_message_sync(body, signature):
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        print(f"LINE handler error: {e}")
        raise HTTPException(status_code=500, detail="LINE handler error")

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    msg = event.message.text.strip()

    #暫停偵測60秒（針對該使用者綁定的所有裝置）
    if msg in ["暫停偵測", "暫停偵測60秒", "暫停", "pause"]:
        db = get_db_connection()
        cur = db.cursor(dictionary=True)
        u = get_user_by_line_id(cur, user_id)
        if not u:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="您尚未加入任何裝置"))
            db.close()
            return
        serials = list_devices_by_user(cur, u['id'])
        for serial in serials:
            pause_device(serial, seconds=60)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="⏸開始暫停偵測60秒"))
        db.close()
        return

    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    # 加入裝置
    if msg.startswith("加入"):
        device_code = msg.replace("加入", "").strip()
        cursor.execute("SELECT id FROM devices WHERE device_id = %s", (device_code,))
        device = cursor.fetchone()
        if not device:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="裝置不存在"))
            db.close()
            return
        # 取得暱稱
        try:
            profile = line_bot_api.get_profile(user_id)
            display_name = profile.display_name
        except Exception:
            display_name = "用戶"

        user = create_user_if_not_exists(db, cursor, user_id, display_name)
        added = map_user_to_device(db, cursor, device['id'], user['id'])
        if added:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"{display_name} 已加入裝置 {device_code}"))
        else:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"你已加入裝置 {device_code}"))
        db.close()
        return

    # 顯示序號登入成員 / 裝置列表
    if msg in ["顯示序號登入成員", "顯示已加入裝置", "裝置列表"]:
        cursor.execute("SELECT id FROM users WHERE line_user_id = %s", (user_id,))
        user = cursor.fetchone()
        if not user:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="您尚未加入任何裝置"))
            db.close()
            return

        cursor.execute("""
            SELECT d.device_id FROM devices d
            JOIN device_user_map m ON d.id = m.device_id
            WHERE m.user_id = %s
        """, (user['id'],))
        devices = cursor.fetchall()
        if not devices:
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="您尚未加入任何裝置"))
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
            """, (device['device_id'],))
            members = cursor.fetchall()
            names = [m['display_name'] or "匿名用戶" for m in members]
            result_msg += f"成員：{', '.join(names)}\n"

        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=result_msg))
        db.close()
        return

    # 今日哭聲紀錄
    if msg in ["3", "3.", "今日哭聲紀錄", "3.今日哭聲紀錄", "3. 今日哭聲紀錄"]:
        logs = get_today_logs_for_user(cursor, user_id)
        if not logs:
            result_msg = "今天還沒有哭聲紀錄！"
        else:
            result_msg = "📊 今天的哭聲紀錄：\n"
            for log in logs:
                reason = log[1] if isinstance(log, tuple) else log.get('reason') or ''
                cry = (log[0] if isinstance(log, tuple) else log.get('cry_type')) or reason or '未知'
                ts = (log[2] if isinstance(log, tuple) else log.get('created_at'))
                hhmm = ts.strftime('%H:%M') if ts else '--:--'
                result_msg += f"- {hhmm}：{cry}\n"

        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=result_msg))
        db.close()
        return
    
    #Template Message
    @handler.add(MessageEvent, message=TextMessage)
    def handle_message(event):
        user_msg = event.message.text.strip()

    if user_msg in ["4", "4.", "寶寶攻略大全", "4.寶寶攻略大全", "4. 寶寶攻略大全"]:
        reply = TemplateSendMessage(
            alt_text='主選單',
            template=ButtonsTemplate(
                title='新生寶寶必修課|爸媽安心看',
                text='請選擇以下功能',
                actions=[
                    MessageAction(
                        label='寶寶必備清單',
                        text='［寶寶必備清單］\n    採購時機建議媽媽在懷孕7至8個月時，先根據自身預算，慢慢列出嬰兒出生用品的採買清單，以免有遺漏，接著在生產前2個月再透過網路或直接前往商場選購即可。\n\n🚼嬰兒車與配件\n	• 嬰兒背帶：可將寶寶背在身上，空出雙手做其他事。建議選擇堅固、易清洗的款式。\n	• 嬰兒推車：方便帶寶寶外出，刺激感官、轉換心情。\n	• 兒童汽車安全座椅：法律規定必備，能保護寶寶安全。注意使用年限（約 5~7 年），避免使用過期或二手產品。\n\n🍼哺乳用品\n	• 餵奶專用枕：減輕手臂與頸部負擔，幫助寶寶維持良好姿勢。\n	• 哺乳小物：綿羊油、冰袋可舒緩脹痛；防溢乳墊避免衣物沾濕。\n	• 圍兜：接住口水或溢奶。\n	• 奶瓶：可選玻璃或不鏽鋼材質，避免化學物質釋出。\n	• 配方奶：母乳不足時可替代，種類多元，建議與醫師討論。\n	• 集乳器：收集乳汁，有手壓型與電動型。\n	• 母乳儲存袋：節省冰箱空間，部分款式可直接解凍使用。\n	• 擠乳組：方便靈活，支援母乳餵養。\n	• 溫奶器：快速解凍或加熱母乳。\n	• 奶瓶刷：大小刷具搭配，清潔更徹底。\n	• 奶瓶清潔劑：專用清潔用品，成分安全。\n	• 奶瓶消毒器：常見有蒸汽、紫外線、化學消毒三種方式。\n	• 奶粉與奶粉攜帶盒：外出或母乳不足時的好幫手，攜帶盒分大小容量方便使用。\n\n🛏新生兒寢具\n	• 毯子：分為可穿式、包巾式，依需求選用。\n	• 嬰兒床與床墊：確保寶寶安全與父母睡眠品質，應選安全檢驗合格、易清洗款式。\n\n🍽餐飲用品\n	• 兒童湯匙：塑膠或橡膠材質，安全又有趣，可吸引寶寶興趣。\n	• 兒童專用碗：幫助寶寶學習自己進食。\n	• 兒童餐椅：附托盤款式方便清潔，也有可拆式托盤可直接洗滌。\n	• 圍兜：防水快乾材質能接住掉落食物，避免用餐過後一團亂。\n\n🧴生活與清潔必備\n	• 嬰兒床或床中床：結構需牢固，避免銳利邊角。\n	• 紙尿布：依寶寶體重挑選尺寸（NB~XXL）。\n	• 固齒器：舒緩長牙不適。\n	• 奶嘴：安撫寶寶情緒，幫助入睡。建議購買 2 個以上替換。\n	• 嬰兒專用指甲剪：避免刮傷肌膚。\n	• 體溫計／耳溫槍：3 個月前建議用體溫計，之後可用耳溫槍。\n	• 嬰兒濕紙巾：選天然溫和配方，呵護肌膚。\n\n🛁盥洗用品\n	• 澡盆：大小適中，方便清洗寶寶。可選折疊式，方便收納。\n	• 嬰兒專用清潔用品：選無香精、無塑化劑的溫和配方。\n	• 嬰兒牙刷：適用於 6 個月後出牙寶寶。\n	• 紗布巾：用途廣，建議準備 5 條以上。\n	• 浴巾：柔軟大尺寸，可包裹全身。\n\n🎲早教或其他\n	• 玩具：建議選擇安全材質，如絨毛玩偶、搖鈴、音樂娃娃。\n	• 童書：啟發語言與想像力，養成閱讀習慣。\n	• 搖搖椅：能隨寶寶動作擺動，並附安全綁帶，讓父母可安心處理其他事。\n\n參考資料：\n- https://www.gbding.com/blog/posts/newborn-baby-essentials-shopping-list#section2\n- https://helloyishi.com.tw/parenting/babys-first-year/baby-care/things-to-buy-for-your-newborn/\n- https://www.runnyyolk.com/blog/posts/2024新手爸媽必看！嬰兒用品必備清單，跟著買就對了'
                    ),
                    MessageAction(
                        label='寶寶照護攻略（抱法、換尿布等）',
                        text='［寶寶照護攻略］\n一、親餵與瓶餵協助：\n  ●親餵協助: 親餵雖然爸爸無法代勞，卻可以從旁協助，讓媽咪的哺乳過程更順暢。\n    1.舒適姿勢: 準備哺乳枕或靠枕，協助太太找到舒適的姿勢開始哺餵。\n    2.防溢吐奶: 準備紗布巾或小毛巾，以防溢吐奶措手不及。\n    3.口腔清潔: 如寶寶清醒狀況許可，喝奶後建議可以用拋棄式紗布沾濕溫開水，幫寶寶清潔口腔。\n    4.按摩舒緩: 可替媽媽按摩頭皮與肩頸，舒緩媽媽情緒。\n  ●瓶餵協助: 媽媽事先擠出來保存的母奶，就可以交給爸爸來進行瓶餵，讓媽媽多出時間做別的事。\n    1.輕點嘴唇: 將奶瓶輕點寶寶嘴唇，此時寶寶會張嘴，將奶嘴完整放入寶寶口中。\n    2.奶瓶傾斜: 奶瓶傾斜使奶水充滿奶嘴，讓寶寶持續吸吮不會吞下空氣。\n    3.暫停拍嗝: 新生兒的飲奶量有限，寶寶不想喝時，可輕輕拍嗝後，再確認是否還想再喝。不要強迫已不想喝奶的嬰兒把剩下的奶喝完。\n    4.口腔清潔: 如寶寶清醒狀況許可，喝奶後建議可以用拋棄式紗布沾濕溫開水，幫寶寶清潔口腔。\n  ★小提醒: 瓶餵時請注意奶水流速，試著將奶瓶倒置，奶水「1秒1滴」的孔洞大小，最適合新生兒吸吮與吞嚥。\n\n二、拍嗝：\n    1.抱好寶寶: 讓寶寶側身坐在大人腿上，爸爸一手從前方托住寶寶的下巴、頸部與肩膀。\n    2.空掌輕拍: 另一手微彎呈空掌，在寶寶背上輕拍。\n\n三、洗澡 ：\n    1.抱好寶寶: 將寶寶以橄欖球姿勢側抱在腋下，讓他枕靠在同側手臂上同時支托住後頸，用大拇指輕按耳朵。\n    2.先洗臉: 另一手拿洗淨擰乾的布巾輕輕擦洗臉部。\n    3.再洗頭: 擠少許洗沐用品，幫寶寶輕輕搓洗頭皮，並以清水沖淨，輕輕擦乾。\n    4.最後身體: 最後下水洗身體，留意後頸、腋下、生殖器等皺褶處比較容易髒污。\n  ★注意事項:\n    ■無論四季，洗澡時室溫維持在25-28℃，最為舒適。\n    ■建議先放冷水再放熱水，確保寶寶安全。\n    ■最適合的洗澡水大約是介於在37-38℃左右，可以用溫度計測量。如寶寶洗完澡後全身紅通通，或是出現起疹發紅現象，表示水太熱，請逐漸調降水溫找到適合的溫度。\n    ■避免寶寶太餓或太飽時洗澡。\n\n四、換尿布：\n    1.洗屁屁: 先用溫清水幫寶寶把屁股洗乾淨(不便時可用濕紙巾)。\n    2.雙腳舉高: 讓寶寶躺好，一手抓住雙腳舉高。\n    3.放尿布: 另一手將乾淨尿布放到適合位置，再將雙腳放下。\n    4.黏好尿布: 撕開尿布兩側的魔鬼氈確實黏好。\n    5.最後檢查: 確認腰部鬆緊度是否合宜，並整理大腿側邊的尿布。\n\n五、臍帶護理：\n    1.先消毒: 滅菌棉棒滴上適量75％酒精，繞著臍帶根部由內往外環狀擦拭一圈。\n    2.後乾燥: 再取另一支滅菌棉棒，滴上適量95％酒精，繞著臍帶根部由內往外環狀擦拭一圈。\n  ★小提醒: 寶寶臍帶照顧的原則是保持乾淨與乾燥，請爸媽們依接生醫院之教導，進行臍帶護理。臍帶未脫落前應讓寶寶穿著寬鬆衣物，並將尿布反摺，使臍部保持乾燥，若有長息肉或是分泌物惡臭、周圍皮膚紅腫等臍帶炎的現象，則請返回醫院診治。\n\n參考資料：\n- https://mammy.hpa.gov.tw/Home/NewsKBContent?id=2403&type=01\n- https://www.cmuh.cmu.edu.tw/HealthEdus/Detail?no=5220#:~:text=新生兒居家注意事項%20*%20先將包布、衣服鋪好，並將大毛巾放置身旁以便取用%E3%80%82%20*%20放冷水再放熱水，以手肘、手腕內側試水溫，以不燙手為原則，溫度40%20℃%20左右%E3%80%82%20*%20餵食後一小時勿洗澡，若天氣寒冷時，選擇一天中氣溫較高的時段為佳%E3%80%82%20*%20頸部、腋下、腹股溝、生殖部都要清洗乾淨%E3%80%82%20*%20可於沐浴後幫寶寶做臍帶護理%E3%80%82\n- https://www.fhs.gov.hk/tc_chi/health_professional/OMP_eNewsletter/enews_20111031.html\n\n延伸資料：\n- 「國民健康署」-母乳哺育手冊 https://reurl.cc/jDKvRm\n- 「社會及家庭署」-ㄜ!我打嗝了 https://reurl.cc/mDK0M7\n- 「社會及家庭署」-寶寶洗澡和清潔 https://reurl.cc/94mR3n\n- 高需求寶寶的照護 https://www.chick.com.tw/baike-detail/育兒資訊站/寶寶居家照顧/high_need_baby?srsltid=AfmBOoroHnD-cSvXzJQ9HnxU_XhanMrHka7-Je5da2NJvulUeVNWMwP_'
                    ),
                    MessageAction(
                        label='寶寶疫苗接種指南',
                        text='［寶寶疫苗接種大全］\nB型肝炎疫苗：出生滿一個月接種。\n卡介苗：建議在出生滿一個月後接種。\n五合一疫苗：包含白喉、破傷風、非細胞性百日咳、b型嗜血桿菌及不活化小兒麻痺。\n13價結合型肺炎鏈球菌疫苗：提供多種肺炎鏈球菌血清型的保護。\n水痘疫苗：預防水痘感染。\n麻疹、腮腺炎、德國麻疹混合疫苗(MMR)：預防三種疾病。\nA型肝炎疫苗：預防A型肝炎感染。\n日本腦炎疫苗：預防日本腦炎感染。\n\n常見副作用：可能的副作用如注射部位局部疼痛或輕微發燒，通常會在接種後幾天內消退。\n\n參考資料：\n- https://nestlebaby.hk/content/0-至2-歲接種疫苗時間表\n- https://kids.heho.com.tw/archives/154606\n- https://www.carloine.com.tw/Article/Detail/78265?gad_source=1&gad_campaignid=19750688572&gbraid=0AAAAACyIkX45uEdr7U6s2XdHIP3_-MZS9&gclid=Cj0KCQjwrJTGBhCbARIsANFBfgvnwu-0E9c5iSS4DX4yn0UW-VY7tTz6pkvlktHPA0nlwHJxNM18kNMaAu3tEALw_wcB'
                    )
                ]
            )
        )
        line_bot_api.reply_message(event.reply_token, reply)