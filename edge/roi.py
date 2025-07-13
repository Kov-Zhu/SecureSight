#!/usr/bin/env python3
import cv2, argparse

def main():
    parser = argparse.ArgumentParser(description="CLI ROI crop")
    parser.add_argument("--input",  "-i", default="full_frame.jpg", help="input image")
    parser.add_argument("--output", "-o", default="roi.jpg",        help="cropped output")
    parser.add_argument("--roi",     "-r", nargs=4, type=int, metavar=('X','Y','W','H'),
                        required=True,
                        help="ROI as x y w h (e.g. 100 50 200 150)")
    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        print(f"ERROR: cannot read {args.input}")
        return

    x, y, w, h = args.roi
    crop = img[y:y+h, x:x+w]
    if crop.size == 0:
        print("ERROR: ROI out of bounds or zero size")
        return

    cv2.imwrite(args.output, crop)
    print(f"Cropped ROI {x,y,w,h} from {args.input} → {args.output}")

if __name__ == "__main__":
    main()
