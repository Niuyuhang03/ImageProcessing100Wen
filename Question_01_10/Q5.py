import numpy as np
import cv2


img = cv2.imread('imori.jpg')
img2 = img.copy().astype(np.float32)

max_v = np.max(img2, axis=2).copy()
min_v = np.min(img2, axis=2).copy()
min_arg = np.argmin(img2, axis=2)
out = np.zeros_like(img2)

S = max_v - min_v
V = max_v

H = np.zeros_like(max_v)
H[np.where(max_v==min_v)] = 0
H[np.where(min_arg==0)] = 60 * (img2[..., 1][np.where(min_arg==0)] - img2[..., 2][np.where(min_arg==0)]) / (max_v[np.where(min_arg==0)] - min_v[np.where(min_arg==0)]) + 60
H[np.where(min_arg==2)] = 60 * (img2[..., 0][np.where(min_arg==2)] - img2[..., 1][np.where(min_arg==2)]) / (max_v[np.where(min_arg==2)] - min_v[np.where(min_arg==2)]) + 180
H[np.where(min_arg==1)] = 60 * (img2[..., 2][np.where(min_arg==1)] - img2[..., 0][np.where(min_arg==1)]) / (max_v[np.where(min_arg==1)] - min_v[np.where(min_arg==1)]) + 300
H = (H + 180) % 360

C = S
H_ = H / 60
X = C * (1 - np.abs( H_ % 2 - 1))
Z = np.zeros_like(H)

vals = [[Z,X,C], [Z,C,X], [X,C,Z], [C,X,Z], [C,Z,X], [X,Z,C]]

for i in range(6):
    ind = np.where((i <= H_) & (H_ < (i+1)))
    out[..., 0][ind] = (V-C)[ind] + vals[i][0][ind]
    out[..., 1][ind] = (V-C)[ind] + vals[i][1][ind]
    out[..., 2][ind] = (V-C)[ind] + vals[i][2][ind]

out[np.where(max_v == min_v)] = 0
out = out.astype(np.uint8)

cv2.imshow('imori', out)
cv2.waitKey(0)
cv2.destroyAllWindows()