import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str)
parser.add_argument('--height', type=int)
parser.add_argument('--width', type=int)
args = parser.parse_args()

img_path = args.img_path
h = args.height
w = args.width

img = cv2.imread(img_path)
img_res = cv2.resize(img, (w, h))

cv2.imwrite(img_path, img_res)
