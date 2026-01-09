import serial
import cv2

ser = serial.Serial("COM8", 115200) #改成Arduino序列埠

def open_camera():
    cap = cv2.VideoCapture(0)  #0=預設攝影機
    if not cap.isOpened():
        print("⚠️無法開啟攝影機")
        return
    print("✅攝影機已開啟(按q關閉)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

while True:
    if ser.in_waiting > 0:
        msg = ser.readline().decode().strip()
        print("收到Arduino訊號：", msg)
        if msg == "CRY":
            open_camera()
#open_camera()