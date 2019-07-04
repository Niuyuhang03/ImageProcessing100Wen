import cv2
import numpy as np


'''Otsu's Method
小于阈值 t 的类记作 0，大于阈值 t 的类记作 1；
w0 和 w1 是被阈值 t 分开的两个类中的像素数占总像素数的比率（满足 w0+w1=1）；
S0^2, S1^2 是这两个类中像素值的方差；
M0, M1 是这两个类的像素值的平均值；
也就是说：
类内方差：Sw^2 = w0 * S0^2 + w1 * S1^2
类间方差：Sb^2 = w0 * (M0 - Mt)^2 + w1 * (M1 - Mt)^2 = w0 * w1 * (M0 - M1) ^2
图像所有像素的方差：St^2 = Sw^2 + Sb^2 = (const)
根据以上的式子，我们用以下的式子计算分离度：  
分离度 X = Sb^2 / Sw^2 = Sb^2 / (St^2 - Sb^2)
也就是说：
argmax_{t} X = argmax_{t} Sb^2
换言之，如果使 Sb^2 = w0 * w1 * (M0 - M1) ^2 最大，就可以得到最好的二值化阈值 t。
'''
img = cv2.imread('imori.jpg')
cv2.imshow('imori', img)
cv2.waitKey(0)

img2 = img.copy().astype(np.float32)
r = img2[..., 2]
g = img2[..., 1]
b = img2[..., 0]
img3 = 0.2126 * r + 0.7152 * g + 0.0722 * b

shape_sum = img3.shape[0] * img3.shape[1]
max_sb = -1
max_t = -1

for t in range(1, 255):
    pixel0 = img3[img3 < t]
    pixel1 = img3[img3 >= t]

    m0 = pixel0.mean() if len(pixel0) > 0 else 0
    m1 = pixel1.mean() if len(pixel1) > 0 else 0

    w0 = len(pixel0) / shape_sum
    w1 = len(pixel1) / shape_sum

    sb_2 = w0 * w1 * pow((m0 - m1), 2)

    if sb_2 > max_sb:
        max_sb = sb_2
        max_t = t

img3[img3 >= max_t] = 255
img3[img3 < max_t] = 0
img4 = img3.astype(np.uint8)
cv2.imshow('imori', img4)
cv2.waitKey(0)
cv2.destroyAllWindows()
