from flask import Flask, render_template, Response
import cv2
import numpy as np

from ultralytics import YOLO
from PIL import Image


app = Flask(__name__)

def generate_frames():
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture("video (2160p).mp4")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame,(1920,1080))
        results = model.predict(frame)
        result = results[0]

        image_cv = cv2.cvtColor(result.plot()[:, :, ::-1], cv2.COLOR_RGB2BGR)

        _, buffer = cv2.imencode('.jpg', image_cv)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
