from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import time
from yolo_utils import preprocess, postprocess, draw_results
from config import MODEL_PATH, INPUT_SIZE, SCORE_THRESHOLD, NMS_THRESHOLD

app = Flask(__name__)

def gen_frames():
    picam2 = Picamera2()
    # Use the default resolution of the camera
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    picam2.start()
    time.sleep(1)
    while True:
        frame = picam2.capture_array()
        orig_h, orig_w = frame.shape[:2]
        # Preprocessing is only scaled for reasoning.
        _, input_tensor, _, _ = preprocess(frame, input_size=INPUT_SIZE)
        # Reasoning
        import onnxruntime as ort
        session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})
        # Post-processing
        results = postprocess(outputs, orig_w, orig_h, conf_threshold=SCORE_THRESHOLD, input_size=INPUT_SIZE)
        # Draw the results on the original frame
        frame = draw_results(frame, results)
        # Encoded as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
      <head>
        <title>Real-time object detection</title>
      </head>
      <body>
        <h1>Real-time object detection</h1>
        <img src="/video" height="640">
      </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)