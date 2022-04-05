from glob import glob
import cv2
import numpy as np

wtky = 30
slp = 0


def birdview(frame):
    rows, cols = frame.shape[0], frame.shape[1]

    w, h = 400,300

    ''' 우리 카메라 기준
    pnt1 = np.int32([[275,320],[365,320],[cols-60, rows-60],[60, rows-60]])
    cv2.polylines(frame,[pnt1],True,(255,255,0),1, cv2.LINE_AA)

    pnt2 = np.int32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    cv2.polylines(frame,[pnt2],True,(255,0,255),1, cv2.LINE_AA)
    '''

    ''' track.avi 기준 '''
    pnt1 = np.int32([[190,280],[450,280],[cols-15, rows-100],[15, rows-100]])
    cv2.polylines(frame,[pnt1],True,(255,255,0),1, cv2.LINE_AA)

    pnt2 = np.int32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    cv2.polylines(frame,[pnt2],True,(255,0,255),1, cv2.LINE_AA)
    ''' ㅡㅡㅡㅡㅡㅡㅡㅡ '''
    
    mtrx = cv2.getPerspectiveTransform(np.float32(pnt1),np.float32(pnt2))

    dst = cv2.warpPerspective(frame, mtrx,(w,h))



def start(img):
    global wtky,slp

    cap = cv2.VideoCapture(img)

    while True:
        ok, frame = cap.read()
        if not ok : break

        dst = birdview(frame)

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