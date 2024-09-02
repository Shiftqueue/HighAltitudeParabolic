import numpy as np


def extract_patch(img, cx, cy, patch_size=50):
    """
    从给定图像和给定中心坐标中截取图像大小为patch_size*patch_size大小的图像，不足之处用黑色填充
    :param img:
    :param cx:
    :param cy:
    :param patch_size:
    :return:
    """
    height, width = img.shape[:2]  # 初始图像大小
    half_patch = patch_size // 2
    left = max(0, cx - half_patch)
    top = max(0, cy - half_patch)
    right = min(width, cx + half_patch)
    bottom = min(height, cy + half_patch)

    left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    cropped = img[top:bottom, left:right]

    background = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    background[0:cropped.shape[0], 0:cropped.shape[1], :] = cropped

    return background


def bbox_diou(box1, box2):
    """
    计算两个矩形框的 DIOU（Distance-IoU）。

    :param box1: 第一个矩形框的坐标 [x1, y1, x2, y2]
    :param box2: 第二个矩形框的坐标 [x1, y1, x2, y2]
    :return: 两个矩形框的 DIOU 值
    """

    # 计算两个矩形框的中心点
    center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]

    # 计算最小包围矩形框
    min_left, min_top = min(box1[0], box2[0]), min(box1[1], box2[1])
    min_right, min_bottom = max(box1[2], box2[2]), max(box1[3], box2[3])

    # 计算最小包围矩形框的对角线距离
    c_width, c_height = min_right - min_left, min_bottom - min_top

    # 计算两个矩形框中心点之间的距离
    distance_center = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    diagonal_length = np.sqrt(c_width ** 2 + c_height ** 2)  # 计算最小包围矩形框的对角线长度

    # 计算 DIOU
    diou = distance_center / diagonal_length
    return 1 - diou

# # 示例
# box_1 = [10, 10, 40, 40]  # [x1, y1, x2, y2]
# box_2 = [70, 70, 110,110 ]  # [x1, y1, x2, y2]
#
# d_iou = bbox_diou(box_1, box_2)
# print("DIOU:", d_iou)
