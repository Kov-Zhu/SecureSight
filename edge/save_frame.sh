#!/usr/bin/env bash
# save_frame.sh
python3 - << 'EOF'
from picamera2 import Picamera2
import time, cv2

picam2 = Picamera2()
cfg = picam2.create_still_configuration(main={"size": (1024, 768)})
picam2.configure(cfg)
picam2.start()
time.sleep(1)
frame = picam2.capture_array()
picam2.stop()

cv2.imwrite("full_frame.jpg", frame)
print("Saved full_frame.jpg (size: {}×{})".format(frame.shape[1], frame.shape[0]))
EOF
