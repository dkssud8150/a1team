import cv2
import os
from glob import glob

video = ["./src/video/track1.avi","./src/video/xycar_track1.mp4","./src/video/base_camera_dark.avi"]
train_cnt = 0
test_cnt = 0

for v in video:
    cap = cv2.VideoCapture(v)

    while True:
        ok, img = cap.read()
        if not ok: break
        os.makedirs("./src/img/track/train/", exist_ok=True)
        os.makedirs("./src/img/track/test/", exist_ok=True)
        
        if len(glob("./src/img/track/train/*")) < 3000:
            cv2.imwrite("./src/img/track/train/%06d.jpg" % train_cnt, img)
            train_cnt += 1
        else:
            cv2.imwrite("./src/img/track/test/%06d.jpg" % test_cnt, img)
            test_cnt += 1


        

    print("finish!")
