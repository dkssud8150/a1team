from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torchvision import models, datasets
import torchvision.transforms as T

wtky = 30
slp = 0


def dplearn(img):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)   
    model.eval()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    img = img[:,:,3]
    pred = model(img)
# https://github.com/dkssud8150/Computer-Vision/blob/main/Object%20Detection/PyTorch_faster_RCNN.ipynb

def start(img, video=True):
    global wtky,slp

    if video:
        cap = cv2.VideoCapture(img)
    else: cap = cv2.imread("./src/img/screen5.png")

    while True:
        if video:
            ok, frame = cap.read()
            if not ok : break
        else: frame = cap

        rows, cols = frame.shape[0], frame.shape[1]

        roi = frame[330:430,110:530]
        cv2.rectangle(frame, (110,330),(530,430),(0,255,255),2)

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(gray_roi, 50, 255, cv2.THRESH_OTSU)[1]

        
        plt.imshow(mask, cmap='gray')
        plt.show()
        cv2.imshow("frame",frame)
#        cv2.imshow("dst",dst)
        if video:
            if cv2.waitKey(wtky) == 27: break
            elif cv2.waitKey(wtky) == ord('s'): wtky = wtky * 2
            elif cv2.waitKey(wtky) == ord('w'): wtky = wtky // 2
        else:
            if cv2.waitKey(0) == 27: break


    
if __name__ == "__main__":
    img_fd = glob("./src/video/*")
    img = "./src/img/screen5.png"

#    for img in img_fd:
#        img = cv2.imread(img_dir)
    start(img,False)
#    start(img_fd[3])