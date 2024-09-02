import cv2
import time
import subprocess
from associate.association_without_model import data_association


def get_v_and_output_v():
    """
    从视频流中读取数据并以视频流的形式发送
    :return: None
    """
    cap = cv2.VideoCapture('put your source address here')  # TODO modify your address
    if not cap.isOpened():
        print("Error opening video file")
        return

    rtsp_url = "the rtsp address"  # TODO modify your address
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_width = 1080
    new_height = int(height * (target_width / int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
    # ffmpeg args
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{target_width}x{new_height}',  # video size
        '-r', str(int(cap.get(cv2.CAP_PROP_FPS))),  # FPS
        '-i', '-',
        '-c:v', 'libx264',  # change to h264
        '-preset', 'ultrafast',
        '-f', 'rtsp',
        rtsp_url
    ]
    # ffmpeg process
    p = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    start_time = time.time()
    frame_count = 0
    frame_rate_period = 1.0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or error capturing frame.")
            break
        if frame is not None:
            frame = cv2.resize(frame, (target_width, new_height), interpolation=cv2.INTER_LINEAR)
            frame_count += 1
            # the core logic function
            data_association(frame)
            elapsed_time = time.time() - start_time
            if elapsed_time >= frame_rate_period:
                start_time = time.time()
                frame_count = 0
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            p.stdin.write(frame.tobytes())
    cap.release()
    p.stdin.close()
    p.wait()


if __name__ == "__main__":
    # TODO
    get_v_and_output_v()
