import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('../data/left.png', 0)
imgR = cv2.imread('../data/right.png', 0)

# SAD window size should be between 5..255
block_size = 15

num_disp = 16    # 必须取16的整数倍
uniquenessRatio = 10


stereo = cv2.StereoBM_create(numDisparities = num_disp, blockSize = block_size)
stereo.setUniquenessRatio(uniquenessRatio)


# disparity = stereo.compute(imgL,imgR)
disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0  


# np.savetxt("../data/disparity_image_bm.txt", disparity, fmt='%3.2f', delimiter=' ', newline='\n')


plt.imshow(disparity,'gray')
plt.axis('off')   # 去掉坐标轴
plt.savefig('...', disparity)
plt.show()

disparity = cv2.imread('', cv2.IMREAD_GRAYSCALE)
disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)    # 转化为伪彩色图
