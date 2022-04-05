#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy, rospkg
import numpy as np
import cv2, random, math, time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import sys
import os
import signal

class PID():
    def __init__(self, kp, ki, kd):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd

        self.p_error = 0.0
        self.i_error = 0.0
        self.d_error = 0.0

    def pid_control(self, cte):

        self.d_error = cte - self.p_error
        self.p_error = cte
        self.i_error += cte

        return self.Kp*self.p_error + self.Ki*self.i_error + self.Kd*self.d_error


def signal_handler(sig, frame):
    os.system('killall -9 python rosout')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

image = np.empty(shape=[0])
bridge = CvBridge()
pub = None
Width = 640
Height = 480
Offset = 400
Gap = 40
past_error=0

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
    center = (lpos + rpos) / 2

    cv2.rectangle(img, (lpos - 5, 15 + offset),
                       (lpos + 5, 25 + offset),
                       (0, 255, 0), 2)
    cv2.rectangle(img, (rpos - 5, 15 + offset),
                       (rpos + 5, 25 + offset),
                       (0, 255, 0), 2)
    cv2.rectangle(img, (center-5, 15 + offset),
                       (center+5, 25 + offset),
                       (0, 255, 0), 2)    
    cv2.rectangle(img, (315, 15 + offset),
                       (325, 25 + offset),
                       (0, 0, 255), 2)
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

        cv2.line(img, (int(x1), Height), (int(x2), (Height/2)), (255, 0,0), 3)

    return img, int(pos)

# show image and return lpos, rpos
def process_image(frame):
    global Width
    global Offset, Gap

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
    all_lines = cv2.HoughLinesP(roi,1,math.pi/180,30,30,10)

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
    cv2.imshow('calibration', frame)

    return lpos, rpos

def s_level(s_value):
    pass

def p_level(p_value):
    pass

def i_level(i_value):
    pass

def d_level(d_value):
    pass

def Offset_level(o_value):
    pass

def Gap_level(g_value):
    pass

cv2.namedWindow("trackbar")
cv2.createTrackbar('s_level','trackbar', 5, 20, s_level)
cv2.createTrackbar('p_level','trackbar', 10, 20, p_level)
cv2.createTrackbar('i_level','trackbar', 0, 100, i_level)
cv2.createTrackbar('d_level','trackbar', 10, 20, d_level)
cv2.createTrackbar('Offset_level','trackbar', 400, 420, d_level)
cv2.createTrackbar('Gap_level','trackbar', 40, 50, d_level)

def start():
    global pub
    global image
    global Width, Height
    global past_error
    global Offset, Gap

    cap = cv2.VideoCapture('xycar_track1.mp4')
    time.sleep(3.0)

    while True:
        ret, image = cap.read()
        while not image.size == (640*480*3):
            continue

        lpos, rpos = process_image(image)

        if (lpos == 0 and rpos < 640):
            e = rpos - Width/2
            lpos -= e
            center = (lpos + rpos) / 2
            error = center - Width / 2

        elif (lpos < 0 and rpos == 640):
            e = Width/2 - lpos
            rpos += e
            center = (lpos + rpos) / 2
            error = center - Width / 2

        elif (lpos == 0 and rpos == 640):
            error = past_error

        else:
            center = (lpos + rpos) / 2
            error = center - Width / 2


        speed = cv2.getTrackbarPos('s_level','trackbar')
        p = cv2.getTrackbarPos('p_level','trackbar') / float(10)
        i = cv2.getTrackbarPos('i_level','trackbar') / float(10000)
        d = cv2.getTrackbarPos('d_level','trackbar') / float(100)
        Offset = cv2.getTrackbarPos('Offset_level','trackbar') 
        Gap = cv2.getTrackbarPos('Gap_level', 'trackbar')

        past_error = error
        pid = PID(p, i, d)
        angle = pid.pid_control(error)

        print "--------"
        print "p_value : {}".format(p)
        print "i_value : {}".format(i)
        print "d_value : {}".format(d)
        #print "lpos : {}".format(lpos)
        #print "rpos : {}".format(rpos)
        #print "angle : {}".format(angle)
        #print "--------"

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':

    start()

