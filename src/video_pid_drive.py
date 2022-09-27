import numpy as np
import cv2, random, math

import sys
import os
import signal

class PID(): 
  def __init__(self,kp,ki,kd):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.p_error = 0.0
    self.i_error = 0.0
    self.d_error = 0.0
    self.angle = 0
    self.past_angle = 0

  def pid_control(self, cte):
    self.d_error = cte-self.p_error
    self.p_error = cte
    self.i_error += cte
    self.past_angle = self.angle
    self.angle = self.kp*self.p_error + self.ki*self.i_error + self.kd*self.d_error
    
    return self.past_angle, self.angle

image = np.empty(shape=[0])

Width = 640
Height = 480
Offset = 410
Gap = 40
speed = 8
past_error=0

    
''' ㅡㅡㅡㅡㅡㅡㅡㅡㅡ이동 평균 필터ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ  '''
# 이동평균 필터 상수
k = 0                               # k번째 수 의미
preAvg = 0                          # 이전의 평균 값
N = 10                             # 슬라이딩 윈도우 크기
c_buf = np.zeros(N + 1)             # 슬라이딩 윈도우

# 이동 평균 필터
def movAvgFilter(pos):
    global k, preAvg, buf, N
    if k == 0:
        buf = pos*np.ones(N + 1)
        k, preAvg = 1, pos
        
    for i in range(0, N):
        buf[i] = buf[i + 1]
    
    buf[N] = pos
    avg = preAvg + (pos - buf[0]) / N
    preAvg = avg
    return int(round(avg))

''' ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ '''

# draw lines
def draw_lines(img, lines):
    global Offset
    for line in lines:
        x1, y1, x2, y2 = line[0]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = cv2.line(img, (x1, y1+Offset), (x2, y2+Offset), color, 2)
    return img

# draw rectangle
def draw_rectangle(img, lpos, rpos, offset=0):
    center = (lpos + rpos) // 2

    cv2.rectangle(img, (lpos - 5, 15 + offset),
                       (lpos + 5, 25 + offset),
                       (0, 255, 0), 2)
    cv2.rectangle(img, (rpos - 5, 15 + offset),
                       (rpos + 5, 25 + offset),
                       (0, 255, 0), 2)
    cv2.rectangle(img, (center - 5, 15 + offset),
                       (center + 5, 25 + offset),
                       (0, 255, 0), 2)
    cv2.rectangle(img, (315, 15 + offset),
                       (325, 25 + offset),
                       (0, 0, 255), 2)
    cv2.rectangle(img, (0,offset),
                       (Width,offset+Gap),
			                 (0,0,0), 3)
    return img

# left lines, right lines
def divide_left_right(lines):
    global Width
    low_slope_threshold = 0
    high_slope_threshold = 10

    # calculate slope & filtering with threshold
    slopes = []
    new_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:
            slope = 0
        else:
            slope = float(y2-y1) / float(x2-x1)
        
        if abs(slope) > low_slope_threshold and abs(slope) < high_slope_threshold:
            slopes.append(slope)
            new_lines.append(line[0])

    # divide lines left to right
    left_lines = []
    right_lines = []

    for j in range(len(slopes)):
        Line = new_lines[j]
        slope = slopes[j]

        x1, y1, x2, y2 = Line

        if (slope < 0) and (x2 < Width/2 - 90):
            left_lines.append([Line.tolist()])
        elif (slope > 0) and (x1 > Width/2 + 90):
            right_lines.append([Line.tolist()])

    return left_lines, right_lines



# get average m, b of lines
def get_line_params(lines):
    # sum of x, y, m
    x_sum = 0.0
    y_sum = 0.0
    m_sum = 0.0

    size = len(lines)
    if size == 0:
        return 0, 0

    for line in lines:
        x1, y1, x2, y2 = line[0]

        x_sum += x1 + x2
        y_sum += y1 + y2
        m_sum += float(y2 - y1) / float(x2 - x1)

    x_avg = x_sum / (size * 2)
    y_avg = y_sum / (size * 2)
    m = m_sum / size
    b = y_avg - m * x_avg

    return m, b

# get lpos, rpos
def get_line_pos(img, lines, left=False, right=False):
    global Width, Height
    global Offset, Gap

    m, b = get_line_params(lines)

    if m == 0 and b == 0:
        if left:
            pos = 0
        if right:
            pos = Width
    else:
        y = Gap / 2
        pos = (y - b) / m

        b += Offset
        x1 = (Height - b) / float(m)
        x2 = ((Height/2) - b) / float(m)

        cv2.line(img, (int(x1), Height), (int(x2), (Height//2)), (255, 0,0), 3)

    return img, int(pos)

# show image and return lpos, rpos
def process_image(frame):
    global Width
    global Offset, Gap
    rows, cols = frame.shape[0], frame.shape[1]

    # gray
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # blur
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

    # canny edge
    low_threshold = 60
    high_threshold = 70
    edge_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold)

    # HoughLinesP
    roi = edge_img[Offset : Offset+Gap, 0 : Width]
    all_lines = cv2.HoughLinesP(roi,1,math.pi/180,20,30,10)

    # divide left, right lines
    if all_lines is None:
        return 0, 640
    left_lines, right_lines = divide_left_right(all_lines)

    # get center of lines
    frame, lpos = get_line_pos(frame, left_lines, left=True)
    frame, rpos = get_line_pos(frame, right_lines, right=True)

    # draw lines
    frame = draw_lines(frame, left_lines)
    frame = draw_lines(frame, right_lines)
    frame = cv2.line(frame, (230, 235), (410, 235), (255,255,255), 2)
                                 
    # draw rectangle
    frame = draw_rectangle(frame, lpos, rpos, offset=Offset)
    #roi2 = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    #roi2 = draw_rectangle(roi2, lpos, rpos)

    # show image
    #cv2.imshow('calibration', frame)

    return lpos, rpos

def start():
    global image
    global cap
    global Width, Height
    global speed, lane_width

    video = ["./src/video/track1.avi","./src/video/road_video1.mp4","./src/video/road_video2.mp4","./src/video/track.avi"]

    cap = cv2.VideoCapture(video[0])

    while True:
        ok,frame = cap.read()
        if not ok : break

        lpos, rpos = process_image(frame)
 
        pid = PID(0.1, 0.0001, 0.07) # p i d
        center = (lpos + rpos)/2
        error = (center - Width/2)
        past_angle, angle = pid.pid_control(error)

        # filter 적용
        lpos, rpos = movAvgFilter(lpos), movAvgFilter(rpos)
    


        print("--------------\n",past_angle, angle,"\n------------") 
        #print("----------------\nlpos : {}, rpos : {}\nerror : {}".format(lpos, rpos, error), "\n----------------")


#        pid = PID(0.45,0.0005,0.05)
#        speed = pid.pid_control(error)

        angle = movAvgFilter(angle)
        print("angle : ", angle, "\nspeed : ", speed)
        
        cv2.imshow("frame",frame)

        if cv2.waitKey(10) == 27: break

if __name__ == '__main__':
    start()
