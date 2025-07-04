#!/usr/bin/env python3
import time
import argparse

# Try to import Picamera2, if it fails, then use OpenCV V4L2 backend.
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

import cv2

def test_with_picamera2(width, height, output):
    """Use Picamera2 to capture frames and save"""
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (width, height)})
    picam2.configure(config)
    picam2.start()
    time.sleep(1)  # wait for camera to warm up
    frame = picam2.capture_array()
    picam2.stop()

    h, w = frame.shape[:2]
    print(f"[Picamera2] Captured frame: {w} x {h}")
    cv2.imwrite(output, frame)
    print(f"[Picamera2] Saved to {output}")

def test_with_opencv(width, height, output):
    """Using OpenCV V4L2 backend to capture frames and save"""
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("ERROR: cannot open camera via V4L2")
        return

    # Set the desired width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: can't receive frame")
    else:
        h, w = frame.shape[:2]
        print(f"[OpenCV] Captured frame: {w} x {h}")
        cv2.imwrite(output, frame)
        print(f"[OpenCV] Saved to {output}")

def main():
    parser = argparse.ArgumentParser(description="Test Raspberry Pi camera")
    parser.add_argument("--width",  type=int, default=640,  help="frame width")
    parser.add_argument("--height", type=int, default=480,  help="frame height")
    parser.add_argument("--output", type=str, default="test_capture.jpg", help="output filename")
    args = parser.parse_args()

    if PICAMERA2_AVAILABLE:
        print("Using Picamera2 backend")
        test_with_picamera2(args.width, args.height, args.output)
    else:
        print("Picamera2 not available, falling back to OpenCV V4L2")
        test_with_opencv(args.width, args.height, args.output)

if __name__ == "__main__":
    main()
