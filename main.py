import cv2,os
import time
from rknnpool import rknnPoolExecutor
# 图像处理函数，实际应用过程中需要自行修改
from func import myFunc

# cap = cv2.VideoCapture('./720p60hz.mp4')
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
cap = cv2.VideoCapture("rtsp://admin:Flink0164199907@192.168.100.84:554/ch_400", cv2.CAP_FFMPEG)
# cap = cv2.VideoCapture(0)
modelPath = "./rknnModel/yolov5s_relu_tk2_RK3588_i8.rknn"
# 线程数, 增大可提高帧率
TPEs = 3
# 初始化rknn池
pool = rknnPoolExecutor(
    rknnModel=modelPath,
    TPEs=TPEs,
    func=myFunc)

# 初始化异步所需要的帧
if (cap.isOpened()):
    for i in range(TPEs + 1):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            del pool
            exit(-1)
        pool.put(frame)

frames, loopTime, initTime = 0, time.time(), time.time()
while (cap.isOpened()):
    frames += 1
    ret, frame = cap.read()
    if not ret:
        break
    pool.put(frame)
    frame, flag = pool.get()
    if flag == False:
        break
    # cv2.imshow('test', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    if not os.path.exists('./result'):
        os.mkdir('./result')
    result_path = os.path.join('./result', 'img_result.jpg')
    cv2.imwrite(result_path, frame)

    if frames % 30 == 0:
        print("30帧平均帧率:\t", 30 / (time.time() - loopTime), "帧")
        loopTime = time.time()

print("总平均帧率\t", frames / (time.time() - initTime))
# 释放cap和rknn线程池
cap.release()
# cv2.destroyAllWindows()
pool.release()
