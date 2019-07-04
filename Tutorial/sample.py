import cv2
import numpy as np

img = cv2.imread("../assets/imori.jpg")
img2 = img.copy().astype(np.float32)
H, W, C = img.shape
img3 = img2[:H // 2, : W // 2, (1, 2, 0)]
cv2.imshow("imori", img2.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()