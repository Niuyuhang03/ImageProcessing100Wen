import cv2
import numpy as np


img = cv2.imread('imori.jpg')
img2 = img.copy().astype(np.float32)
b = img2[:, :, 0]
g = img2[:, :, 1]
r = img2[:, :, 2]
# Y = 0.2126 R + 0.7152 G + 0.0722 B
img3 = 0.2126 * r + 0.7152 * g + 0.0722 * b
cv2.imshow('imori', img3.astype(np.uint8))
cv2.waitKey(0)
cv2.destoryAllWindows()
