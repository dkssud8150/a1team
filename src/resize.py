from glob import glob
import cv2
import numpy as np

def resize(frame):
    rows, cols = frame.shape[0], frame.shape[1]

    ''' 크기 줄여서 연산 후 키워서 연산하면 연산속도 상승 여부 --> 너무 많이 깨짐'''
    frame = cv2.resize(frame,(int(cols*0.2),int(rows*0.2)),interpolation=cv2.INTER_AREA)

    # gray
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # blur
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

    # canny edge
    low_threshold = 60
    high_threshold = 70
    edge_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold)
    
    edge_img = cv2.resize(edge_img,(int(frame.shape[1]*5),int(frame.shape[0]*2)),interpolation=cv2.INTER_LINEAR)

def start(img):
    global wtky,slp

    cap = cv2.VideoCapture(img)

    while True:
        ok, frame = cap.read()
        if not ok : break

        dst = resize(frame)

        cv2.imshow("frame",frame)
        cv2.imshow("dst",dst)
        if cv2.waitKey(wtky) == 27: break
        elif cv2.waitKey(wtky) == ord('s'): wtky = wtky * 2
        elif cv2.waitKey(wtky) == ord('w'): wtky = wtky // 2


if __name__ == "__main__":
    img_fd = glob("./src/video/*")

#    for img in img_fd:
#        img = cv2.imread(img_dir)
#        start(img)
    start(img_fd[3])