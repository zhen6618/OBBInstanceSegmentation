import os
import glob
import numpy as np
import cv2

# 指定你想要搜索的目录
directory = './'

# 使用glob模块匹配所有.png文件
png_files = glob.glob(os.path.join(directory, 'cam_demo*'))

imgs = np.zeros((1024, 1024, 3))

# 输出所有.png文件的文件名
for filename in png_files:
    img = cv2.imread(filename)
    imgs += img

imgs /= len(png_files)
imgs = imgs.astype(np.uint8)
cv2.imwrite('cam_fuse.png', imgs)

print('')