import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('/home/hasantha/Desktop/repos/old-yolov4-deepsort-master/data/download2.png' ,0)
#img=img[423:998,806:1408]

ret, bw_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY) #165
kernel = np.ones((1,1),np.uint8)
#erosion = cv2.erode(img,kernel,iterations = 1)
opening = cv2.morphologyEx(bw_img, cv2.MORPH_OPEN, kernel)
#closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

#Append lane line coordinates to a matrix
lane_line_co=[]
x_cord_lane = np.where(opening == 255)[1]#+806     #+805
y_cord_lane = np.where(opening == 255)[0]#+423     #+389

for q in range(0, len(x_cord_lane)):
    lane_line_co.append((x_cord_lane[q],y_cord_lane[q]))


def get_lane_line_co():
    return x_cord_lane,y_cord_lane,lane_line_co

#print(lane_line_co)

#print(get_lane_line_co()[2])