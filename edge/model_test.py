import cv2
import numpy as np
import onnxruntime as ort
import time
from yolo_utils import preprocess, postprocess, draw_results
from config import TEST_IMG_PATH, MODEL_PATH, TEST_RESULT_PATH

def main():
    print("[INFO] Loading model...")
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    print("[INFO] Preprocessing image...")
    image, input_tensor, orig_w, orig_h = preprocess(TEST_IMG_PATH)

    print("[INFO] Running inference...")
    start = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    latency = (time.time() - start) * 1000
    print(f"[INFO] Inference done in {latency:.2f} ms")

    print("[INFO] Postprocessing...")
    results = postprocess(outputs, orig_w, orig_h)

    print(f"[INFO] {len(results)} object(s) detected")
    output_image = draw_results(image, results)
    cv2.imwrite(TEST_RESULT_PATH, output_image)
    print(f"[INFO] Result saved to {TEST_RESULT_PATH}")

if __name__ == "__main__":
    main()
