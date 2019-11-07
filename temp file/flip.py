import cv2
import os
from PIL import Image
import numpy as np

path = './train_images'
num = 200

# 反転プログラム
# for o in os.listdir(path):
#   image_path = os.path.join(path, o)
#   image2 = Image.open(image_path)
#   image3 = np.array(image2, 'uint8')
#   image4 = cv2.flip(image3, 1)
#   image5 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)
#   filename = str(num) + '.jpg'     
#   cv2.imwrite(filename, image5)
#   num += 1


# 回転プログラム
for o in os.listdir(path):
  image_path = os.path.join(path, o)
  image2 = Image.open(image_path)
  image3 = image2.rotate(45)
  image4 = np.array(image3, 'uint8')
  image5 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)
  filename = str(num) + '.jpg'     
  cv2.imwrite(filename, image5)
  num += 1

  image6 = image2.rotate(315)
  image7 = np.array(image6, 'uint8')
  image8 = cv2.cvtColor(image7, cv2.COLOR_BGR2RGB)
  filename = str(num) + '.jpg'     
  cv2.imwrite(filename, image8)
  num += 1