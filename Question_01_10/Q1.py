import cv2


img = cv2.imread("imori.jpg")
img2 = img[..., (2, 1, 0)]
cv2.imshow('imori', img2)
cv2.waitKey(0)