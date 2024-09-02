import cv2
import yaml
from sklearn.cluster import MeanShift

with open('./conf/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

MIN_DETECT_OBJECT = config['background_subtractor']['min_detect_object']

kernel_size = (config['kernel_size']['width'], config['kernel_size']['height'])
bg_subtract_params = {
    'history': config['background_subtractor']['history'],
    'varThreshold': config['background_subtractor']['var_threshold'],
    'detectShadows': config['background_subtractor']['detect_shadows']
}

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
bg_subtract = cv2.createBackgroundSubtractorMOG2(**bg_subtract_params)


def detect_targets(image):
    """
    基于高斯背景建模的目标检测器
    :param image: 当前帧
    :return: 返回当前画面中主要移动的物体中心坐标
    """
    targets = []
    image = bg_subtract.apply(image)  # 背景建模
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)  # 去除噪点
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
    # 给出符合阈值的轮廓
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        if perimeter > MIN_DETECT_OBJECT:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
            targets.append([cx, cy])

    if len(targets) == 0:
        return targets

    ms = MeanShift(max_iter=100)
    ms.fit(targets)
    cluster_centers = ms.cluster_centers_
    return cluster_centers
