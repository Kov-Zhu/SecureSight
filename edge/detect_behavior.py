#!/usr/bin/env python3
import os
import time

import cv2
import numpy as np
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter

# —— Configuration —— #
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
ROI_BOX       = (100, 50, 200, 150)
MODEL_PATH    = "models/behavior_detector.tflite"
SCORE_THRESHOLD = 0.5
OUTPUT_DIR    = "alerts"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# —————————————— #

def load_model(model_path):
    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()

def preprocess_image(image, input_detail):
    _, in_h, in_w, _ = input_detail['shape']
    # OpenCV returns BGR, it needs to be converted to RGB.
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (in_w, in_h))
    dtype = input_detail['dtype']
    if dtype == np.uint8:
        scale, zero_point = input_detail['quantization']
        norm = resized.astype(np.float32) / 255.0
        quant = norm / scale + zero_point
        return np.expand_dims(quant.astype(np.uint8), axis=0)
    else:
        norm = resized.astype(np.float32) / 255.0
        return np.expand_dims(norm, 0)

def run_inference(interpreter, inp_det, out_det, inp_data):
    # Note: The output order of the COCO SSD model is [boxes, classes, scores, num]
    classes_idx = out_det[1]['index']
    scores_idx  = out_det[2]['index']
    boxes_idx   = out_det[0]['index']

    interpreter.set_tensor(inp_det[0]['index'], inp_data)
    start = time.time()
    interpreter.invoke()
    latency = (time.time() - start) * 1000

    boxes  = interpreter.get_tensor(boxes_idx)[0]   # [[ymin, xmin, ymax, xmax], ...]
    scores = interpreter.get_tensor(scores_idx)[0]  # [score1, score2, ...]
    return boxes, scores, latency

def capture_frame():
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    picam2.configure(cfg)
    picam2.start()
    time.sleep(1)
    raw = picam2.capture_array()  # HxWx4 (XBGR8888)
    picam2.stop()
    # Drop the 4th channel to get HxWx3 BGR
    frame = raw[:, :, :3]
    frame = np.ascontiguousarray(frame)
    return frame

def load_labels(label_path):
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def main():
    print(f"Loading model: {MODEL_PATH}")
    interpreter, inp_det, out_det = load_model(MODEL_PATH)
    labels = load_labels("models/labelmap.txt")

    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    picam2.configure(cfg)
    picam2.start()
    time.sleep(1)

    try:
        while True:
            raw = picam2.capture_array()
            frame = raw[:, :, :3]
            frame = np.ascontiguousarray(frame)
            x, y, w, h = ROI_BOX
            roi = frame[y:y+h, x:x+w]
            inp_data = preprocess_image(roi, inp_det[0])
            boxes, scores, latency = run_inference(interpreter, inp_det, out_det, inp_data)
            classes_idx = out_det[1]['index']
            classes = interpreter.get_tensor(classes_idx)[0]
            for i, score in enumerate(scores):
                if score >= SCORE_THRESHOLD:
                    y1, x1, y2, x2 = boxes[i]
                    x1 = int(x1 * w) + x; y1 = int(y1 * h) + y
                    x2 = int(x2 * w) + x; y2 = int(y2 * h) + y
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                    class_id = int(classes[i])
                    label = labels[class_id] if class_id < len(labels) else str(class_id)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.imshow('Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
