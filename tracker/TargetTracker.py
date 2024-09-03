import cv2
import numpy as np
import math
from datetime import datetime
import yaml

with open('./conf/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
MIN_SHOW_NUM = config['image']['min_show_num']
SAVE_PATH = config['save']['save_path']


def get_color(hierarchy):
    """
    根据等级返回颜色，用于轨迹绘制
    :param hierarchy: 等级
    :return: 轨迹颜色
    """
    if hierarchy >= 2:
        return 0, 0, 255
    color_map = [(122, 122, 122), (0, 0, 255)]
    return color_map[hierarchy]


class TargetTracker:
    """
    使用了卡尔曼滤波器的目标追踪器，预留图片相似度计算
    """

    def __init__(self, center):
        self.kalman_filter = cv2.KalmanFilter(4, 2)
        self.kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                        [0, 1, 0, 1],
                                                        [0, 0, 1, 0],
                                                        [0, 0, 0, 1]], np.float32)
        self.kalman_filter.measurementMatrix = 1.0 * np.eye(2, 4, dtype=np.float32)  # 设置测量噪声协方差矩阵
        self.kalman_filter.processNoiseCov = 1e-5 * np.eye(4, 4, dtype=np.float32)  # 设置过程噪声协方差矩阵
        self.kalman_filter.measurementNoiseCov = 1e-3 * np.eye(2, 2, dtype=np.float32)  # 设置测量噪声协方差矩阵
        self.kalman_filter.errorCovPost = 1.0 * np.eye(4, 4, dtype=np.float32)  # 设置估计误差协方差矩阵
        self.kalman_filter.statePost = np.array([[0], [0], [0], [0]], np.float32)  # 初始化状态向量
        # 初始化位置与运动信息
        self.kalman_filter.statePre = np.array([[center[0]], [center[1]], [0], [0]], np.float32)
        self.kalman_filter.statePost = np.array([[center[0]], [center[1]], [0], [0]], np.float32)
        self.history = []  # 存储历史位置
        self.target_max_confidence = 0.0  # 锁定最大可信的目标
        self.target_missed_frames = 0  # 连续未检测到帧数
        self.target_tracked_frames = 1  # 已经追踪的帧数
        self.is_save = False  # 是否达到保存条件
        self.feature = None  # 保存特征
        self.appearance_update_frequency = 0  # 目标图像定期更新
        self.object_color = (0, 0, 0)  # 不同的配置颜色以醒目的标识
        self.create_time = datetime.now()  # 记录初次探测时间

    def update(self, measurement):
        """
        更新卡尔曼滤波器预测，并更新相关参数
        :param measurement: 新的目标中心点坐标
        :return:
        """
        self.target_missed_frames = 0  # 重置连续未检测帧数
        self.target_tracked_frames += 1
        self.appearance_update_frequency += 1
        self.kalman_filter.predict()  # 预测下一步的状态
        self.kalman_filter.correct(measurement)  # 更新状态向量
        # 保存当前状态估计的位置部分
        position = (int(measurement[0]), int(measurement[1]))
        self.history.append(position)

    def get_prediction(self):
        """
        返回卡尔曼滤波器下一步的预测状态
        :return: 下一步的预测状态
        """
        return self.kalman_filter.predict()  # 预测下一步的状态

    def track_save_or_not(self):
        """
        判断正在追踪的这条轨迹是否能够保留，此函数方法粗略
        TODO 需要考虑什么样的轨迹需要保留
        :return: 判断结果
        """
        if self.is_save:
            return True
        self.object_color = get_color(0)
        if len(self.history) < MIN_SHOW_NUM:
            return False
        # 计算 x 和 y 值的最大值和最小值
        min_x = min([point[0] for point in self.history])
        max_x = max([point[0] for point in self.history])
        min_y = min([point[1] for point in self.history])
        max_y = max([point[1] for point in self.history])
        # 计算矩形的宽度和高度
        width = max_x - min_x
        height = max_y - min_y
        # 使用勾股定理计算对角线长度
        diagonal_length = math.sqrt(width ** 2 + height ** 2)
        if diagonal_length >= 200:
            self.is_save = True
            self.object_color = get_color(1)
            return True
        return False

    # 轨迹线图片
    def draw_line_track2pic(self, image):
        """
        在实时图像中绘制轨迹
        :param image: 当前帧
        :return: None
        """
        if len(self.history) == 0:
            return
        last_point = None
        b, g, r = self.object_color
        p, q = self.history[-1]
        width, height = 15, 15
        # 使用 cv2.rectangle 绘制矩形
        cv2.rectangle(image, (int(p) - width, int(q) - height), (int(p + width), int(q + height)), (b, r, g), 1)
        for x, y in reversed(self.history):
            if last_point is not None:
                cv2.line(image, last_point, (int(x), int(y)), (b, g, r), 2)
            last_point = (int(x), int(y))
        if self.is_save:
            cv2.putText(image, f"{self.create_time.strftime("%Y%m%d%H%M%S")}", (int(p) - width, int(q) - height),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def save_line_track2pic(self, image):
        """
        保存检测到的轨迹至指定位置，路径须在配置文件中更改
        :param image: 当前帧
        :return: None
        """
        if not self.is_save:
            return
        save_image = np.copy(image)
        cv2.putText(save_image, f"{self.create_time.strftime("%Y-%m-%d %H:%M:%S")}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        p, q = self.history[-1]
        width, height = 15, 15
        cv2.rectangle(save_image, (int(p) - width, int(q) - height), (int(p + width), int(q + height)), (0, 255, 0), 1)

        last_point = None
        for x, y in reversed(self.history):
            if last_point is not None:
                cv2.line(save_image, last_point, (int(x), int(y)), (0, 0, 225), 2)
            last_point = (int(x), int(y))
        cv2.putText(save_image, f"{self.create_time.strftime("%Y%m%d%H%M%S")}", (int(p) - width, int(q) - height),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imwrite('./save/' + f"{self.create_time.strftime("%Y%m%d_%H%M%S")}.jpg", save_image)
