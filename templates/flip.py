import cv2
import os

path = './train_images'
num = 0

for o in os.listdir(path):
  image_path = os.path.join(path, o)
  image2 = Image.open(image_path).flip
  filename = str(num) + ',jpg'     
  cv2.imwrite('./train2', filename)
  num += 1

