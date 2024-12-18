import cv2
import numpy as np
from matplotlib import pyplot as plt


def canny_edge_detection(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用高斯模糊减少噪声
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 应用Canny边缘检测
    edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

    # 显示原始图像和边缘检测结果
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray'), plt.title('Edge Image')
    plt.xticks([]), plt.yticks([])
    plt.show()


# 使用示例
canny_edge_detection('portrait.jpg')



