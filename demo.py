import cv2
import time
from associate.association_without_model import data_association


def process_video_stream():
    # 视频流地址
    video_source = 'data/video/000.mp4'
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error opening video stream. Retrying...")
    start_time = time.time()  # 开始计时
    frame_count = 0  # 初始化帧计数器
    frame_rate_period = 1.0  # 设置 FPS 计算周期为 1 秒
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or error capturing frame.")
            break
        if frame is not None:
            height, _, _ = frame.shape
            target_width = 1080
            new_height = int(height * (target_width / frame.shape[1]))
            frame = cv2.resize(frame, (target_width, new_height), interpolation=cv2.INTER_LINEAR)
            frame_count += 1
            frame = data_association(frame)
            # 每隔一定时间重置计数器
            elapsed_time = time.time() - start_time
            if elapsed_time >= frame_rate_period:
                start_time = time.time()
                frame_count = 0
            # 计算并显示FPS
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.2f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('pic', frame)
            # 添加等待键输入
            if cv2.waitKey(10) & 0xFF == ord('q'):  # 按 'q' 键退出
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video_stream()
