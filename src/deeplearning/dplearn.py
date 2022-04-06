from glob import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import models, datasets
import torchvision.transforms as T

wtky = 30
slp = 0

COCO_INSTANCE_CATEGORY_NAMES = []

import matplotlib.pyplot as plt

def model_train(model,epoch = 10):
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    os.makedirs("./weight",exist_ok=True)

    torch.save(model, "./weight/model.pt")
    dummy_input = torch.randn(1, 3, 224, 224,dtype=torch.float32)
    torch.onnx.export(model, dummy_input, "./weight/model.onnx",export_params=True)


def img_prediction(img):
    model = torch.load("./src/deeplearning/weight/model.pt")
    model.eval()

    #print(model.head)

    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    

    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    
    return pred, COCO_INSTANCE_CATEGORY_NAMES


def get_prediction(img, threshold=0.5):
    pred, COCO_INSTANCE_CATEGORY_NAMES = img_prediction(img)

    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    if pred_score[0] < threshold:
        threshold = pred_score[0] - 0.0001
    pred_thres = [pred_score.index(x) for x in pred_score if x>threshold][-1]
    pred_boxes = pred_boxes[:pred_thres+1]
    pred_class = pred_class[:pred_thres+1]

    return pred_boxes, pred_class


def object_detection_plt(img, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    boxes, pred_cls = get_prediction(img, threshold)
    #print(boxes)
    for i in range(len(boxes)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # rectangle 사각형 그리기 함수, 시작점 좌표와 종료점 좌표를 기입하면 직각 사각형을 그릴 수 있다.
    cv2.putText(img,pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # puttext 문자 기입 함수, 글자 우측 하단을 시작점으로 하여 주어진 텍스트를 출력 
    plt.figure(figsize=(20,30))
    plt.imshow(img)
    plt.show()


width = 640
height = 480

window_title = 'camera'

warp_img_w = 320 # w * 1/2
warp_img_h = 240 # h * 1/2

warpx_margin = 20
warpy_margin = 3

nwindows = 9
margin = 12
minpixel = 5
lane_bin_th = 145


warp_src  = np.array([
    [230-warpx_margin, 300-warpy_margin],  
    [45-warpx_margin,  450+warpy_margin],
    [445+warpx_margin, 300-warpy_margin],
    [610+warpx_margin, 450+warpy_margin]
], dtype=np.float32)

warp_dst = np.array([
    [0,0],
    [0,warp_img_h],
    [warp_img_w,0],
    [warp_img_w, warp_img_h]
], dtype=np.float32)


calibrated = True
if calibrated:
    mtx = np.array([
        [422.037858, 0.0, 245.895397], 
        [0.0, 435.589734, 163.625535], 
        [0.0, 0.0, 1.0]
    ])
    dst = np.array([-0.289296, 0.061035, 0.001786, 0.015238, 0.0])
    cal_mtx, cal_roi = cv2.getOptimalNewCameraMatrix(mtx, dst, (width, height), 1, (width, height))

def calibrate_image(frame):
    global width, height
    global mtx, dst
    global cal_mtx, cal_roi
    
    tf_image = cv2.undistort(frame, mtx, dst, None, cal_mtx)
    x, y, w, h = cal_roi
    tf_image = tf_image[y:y+h, x:x+w]

    return cv2.resize(tf_image, (width, height))

def warp_image(img, src, dst, size):
    m_to_dst = cv2.getPerspectiveTransform(src, dst)
    m_to_src = cv2.getPerspectiveTransform(dst, src)
    warp_img = cv2.warpPerspective(img, m_to_dst, size, flags=cv2.INTER_LINEAR)

    return warp_img, m_to_dst, m_to_src

def warp_process_image(img):
    global nwindows
    global margin
    global minpixel
    global lane_bin_th


    blur = cv2.GaussianBlur(img,(5, 5), 0)
    _, L, _ = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2HLS))
    _, lane = cv2.threshold(L, lane_bin_th, 255, cv2.THRESH_BINARY)
    histogram = np.sum(lane[lane.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    leftx_current = np.argmax(histogram[:midpoint])
    rightx_current = np.argmax(histogram[midpoint:]) + midpoint
    window_height = np.int(lane.shape[0]/nwindows)
    nz = lane.nonzero()

    left_lane_inds = []
    right_lane_inds = []
    
    lx, ly, rx, ry = [], [], [], []

    out_img = np.dstack((lane, lane, lane))*255

    for window in range(nwindows):
        win_yl = lane.shape[0] - (window+1)*window_height
        win_yh = lane.shape[0] - window*window_height

        win_xll = leftx_current - margin
        win_xlh = leftx_current + margin
        win_xrl = rightx_current - margin
        win_xrh = rightx_current + margin

        cv2.rectangle(out_img,(win_xll,win_yl),(win_xlh,win_yh),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xrl,win_yl),(win_xrh,win_yh),(0,255,0), 2)

        good_left_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xll)&(nz[1] < win_xlh)).nonzero()[0]
        good_right_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xrl)&(nz[1] < win_xrh)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpixel:
            leftx_current = np.int(np.mean(nz[1][good_left_inds]))
        if len(good_right_inds) > minpixel:        
            rightx_current = np.int(np.mean(nz[1][good_right_inds]))

        lx.append(leftx_current)
        ly.append((win_yl + win_yh)/2)
        rx.append(rightx_current)
        ry.append((win_yl + win_yh)/2)

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    lfit = np.polyfit(np.array(ly),np.array(lx),2)
    rfit = np.polyfit(np.array(ry),np.array(rx),2)

    out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
    out_img[nz[0][right_lane_inds] , nz[1][right_lane_inds]] = [0, 0, 255]
    cv2.imshow("viewer", out_img)

    return lfit, rfit

def draw_lane(image, warp_img, m_to_src, left_fit, right_fit):
    global width, height
    ymax = warp_img.shape[0]
    ploty = np.linspace(0, ymax - 1, ymax)

    # 29.
    color_warp = np.zeros_like(warp_img).astype(np.uint8)

    # 30.
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # 31.
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))]) 
    pts = np.hstack((pts_left, pts_right))

    # 31.
    color_warp = cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, m_to_src, (width, height))

    return cv2.addWeighted(image, 1, newwarp, 0.3, 0)




def video_start(img):
    global wtky,slp

    cap = cv2.VideoCapture(img)

    while True:
        ok, frame = cap.read()
        if not ok : break

        rows, cols = frame.shape[0], frame.shape[1]

        boxes, pred_cls = get_prediction(frame)

        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, pred_cls = get_prediction(rgb_img)

        ''' object mask '''
        for box in boxes:
            cv2.rectangle(frame, (int(box[0][0]),int(box[0][1])), (int(box[1][0]),int(box[1][1])), color=(0,0,0), thickness=-1)

        image = calibrate_image(frame)
        warp_img, m_to_dst, m_to_src = warp_image(image, warp_src, warp_dst, (warp_img_w, warp_img_h))
        left_fit, right_fit = warp_process_image(warp_img)
        lane_img = draw_lane(image, warp_img, m_to_src, left_fit, right_fit)

        cv2.imshow("frame",frame)
#        cv2.imshow("dst",dst)
        
        if cv2.waitKey(wtky) == 27: break
        elif cv2.waitKey(wtky) == ord('s'): wtky = wtky * 2
        elif cv2.waitKey(wtky) == ord('w'): wtky = wtky // 2
        

def img_start(img):
    cap = cv2.imread(img)

    rgb_img = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
    boxes, pred_cls = get_prediction(rgb_img)
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    ''' object mask '''
    for box in boxes:
        cv2.rectangle(cap, (int(box[0][0]),int(box[0][1])), (int(box[1][0]),int(box[1][1])), color=(0,0,0), thickness=-1)
    


    cv2.imshow("src",bgr_img)
    cv2.imshow("cap",cap)
    if cv2.waitKey(0) == 27: return 0
    


if __name__ == "__main__":
    for img in glob("./src/video/*"):
        video_start(img)
#    for img in glob("./src/img/*.png"):
#        img_start(img)
