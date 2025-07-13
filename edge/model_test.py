import cv2
import numpy as np
import onnxruntime as ort
import time

MODEL_PATH = "models/yolo11n.onnx"
IMG_PATH = "test.jpg"
RESULT_PATH = "result.jpg"
INPUT_SIZE = 640  # 根据你的模型定义

# 预处理图像
def preprocess(image_path):
    image = cv2.imread(image_path)
    orig_h, orig_w = image.shape[:2]
    resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = rgb.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return image, img.astype(np.float32), orig_w, orig_h

# 计算 IoU 用于 NMS
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

# 非极大值抑制
def nms(boxes, iou_threshold=0.4):
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    filtered = []

    while boxes:
        chosen = boxes.pop(0)
        filtered.append(chosen)
        boxes = [b for b in boxes if compute_iou(chosen[:4], b[:4]) < iou_threshold]

    return filtered

def postprocess(outputs, orig_w, orig_h, conf_threshold=0.4):
    predictions = outputs[0]  # (1, 84, 8400)
    predictions = np.transpose(predictions, (0, 2, 1))  # -> (1, 8400, 84)
    predictions = predictions[0]  # -> (8400, 84)

    results = []
    for det in predictions:
        x, y, w, h = det[0:4]
        class_scores = det[4:]
        class_id = np.argmax(class_scores)
        conf = class_scores[class_id]

        if class_id != 0 or conf < conf_threshold:
            continue

        x1 = int((x - w / 2) / INPUT_SIZE * orig_w)
        y1 = int((y - h / 2) / INPUT_SIZE * orig_h)
        x2 = int((x + w / 2) / INPUT_SIZE * orig_w)
        y2 = int((y + h / 2) / INPUT_SIZE * orig_h)

        results.append((x1, y1, x2, y2, conf))

    return nms(results)


# 绘图
def draw_results(image, results):
    for (x1, y1, x2, y2, score) in results:
        label = f"person: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# 主程序
def main():
    print("[INFO] Loading model...")
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    print("[INFO] Preprocessing image...")
    image, input_tensor, orig_w, orig_h = preprocess(IMG_PATH)

    print("[INFO] Running inference...")
    start = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    latency = (time.time() - start) * 1000
    print(f"[INFO] Inference done in {latency:.2f} ms")

    print("[INFO] Postprocessing...")
    results = postprocess(outputs, orig_w, orig_h)

    print(f"[INFO] {len(results)} object(s) detected")
    output_image = draw_results(image, results)
    cv2.imwrite(RESULT_PATH, output_image)
    print(f"[INFO] Result saved to {RESULT_PATH}")

if __name__ == "__main__":
    main()
