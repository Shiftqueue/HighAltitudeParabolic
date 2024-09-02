import cv2
import numpy as np
from tracker.TargetTracker import TargetTracker
from target.Targets import detect_targets
from utils.box import bbox_diou
import yaml

with open('./conf/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# 必要参数
MAX_MISSED_FRAMES = config['kalman_filter']['max_missed_frames']
MAX_TRACKED_FRAMES = config['kalman_filter']['max_tracked_frames']
MAX_KF_NUM = config['kalman_filter']['max_filter_count']
MIN_CONFIDENCE = config['kalman_filter']['min_confidence']
IMG_SIZE = config['image']['size']

# 初始跟踪状态（无目标，无跟踪）
kalman_filters = []


def data_association(image):
    """
    数据关联器
    :param image: 当前帧
    :return: None
    """
    global kalman_filters
    # 移除过时的跟踪器
    to_remove = []
    for i in range(len(kalman_filters)):
        if (kalman_filters[i].target_missed_frames >= MAX_MISSED_FRAMES or
                kalman_filters[i].target_tracked_frames >= MAX_TRACKED_FRAMES):
            kalman_filters[i].save_line_track2pic(image)
            to_remove.append(i)
    for i in reversed(to_remove):
        del kalman_filters[i]

    # 获取目标列表
    target_list = detect_targets(image)

    if len(target_list) > 0 and len(kalman_filters) > 0:
        # 关联矩阵
        associate_matrix = np.zeros((len(kalman_filters), len(target_list)))

        for tg_id, tg in enumerate(target_list):
            cx, cy = tg  # 当前目标中心点坐标
            tg_box = [cx - IMG_SIZE // 2, cy - IMG_SIZE // 2,
                      cx + IMG_SIZE // 2, cy + IMG_SIZE // 2]
            for kf_id, kf in enumerate(kalman_filters):
                kf_pre_center = kf.get_prediction()
                kf_box = [kf_pre_center[0] - IMG_SIZE // 2, kf_pre_center[1] - IMG_SIZE // 2,
                          kf_pre_center[0] + IMG_SIZE // 2, kf_pre_center[1] + IMG_SIZE // 2]
                diou = bbox_diou(kf_box, tg_box)
                confidence = diou
                associate_matrix[kf_id][tg_id] = confidence

        # 扫描矩阵，按照最大可信度分配目标
        for kf_id in range(len(associate_matrix)):
            max_value_i = np.max(associate_matrix[kf_id, :])
            if max_value_i < MIN_CONFIDENCE:
                kalman_filters[kf_id].target_missed_frames += 1
                continue
            max_index_i = np.argmax(associate_matrix[kf_id, :])
            cx, cy = target_list[max_index_i]
            kalman_filters[kf_id].update(np.array([[np.float32(cx)], [np.float32(cy)]]))  # 更新参数
            kalman_filters[kf_id].track_save_or_not()  # 判断路径是否保留

        for tg_id in range(len(associate_matrix[0])):
            max_value_i = np.max(associate_matrix[:, tg_id])
            if max_value_i < MIN_CONFIDENCE:
                if len(kalman_filters) >= MAX_KF_NUM:
                    continue
                cx, cy = target_list[tg_id]
                kf = TargetTracker((cx, cy))
                kalman_filters.append(kf)

    elif len(kalman_filters) > 0:
        for _, kf in enumerate(kalman_filters):
            kf.target_missed_frames += 1
    else:
        for _, tg in enumerate(target_list):
            if len(kalman_filters) >= MAX_KF_NUM:
                continue
            cx, cy = tg
            kf = TargetTracker((cx, cy))
            kalman_filters.append(kf)

    # 绘制
    for i, kf in enumerate(kalman_filters):
        kf.draw_line_track2pic(image)
    for cx, cy in target_list:
        cv2.circle(image, (int(cx), int(cy)), 4, (255, 255, 255), -1)
    return image
