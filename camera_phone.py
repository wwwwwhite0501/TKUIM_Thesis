import cv2
from flask import Flask, Response

app = Flask(__name__)

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            #串流回傳
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>手機請用瀏覽器開啟: http://[電腦IP]:5000/video_feed</h1>"

if __name__ == '__main__': #Flask預設的網頁伺服器通訊埠號碼
    app.run(host='0.0.0.0', port=5000)