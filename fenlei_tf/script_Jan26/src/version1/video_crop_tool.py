#originally by Hamed, 25Apr.2016
#hy: this tool is also integrated in tensor_cnn_video.py

import settings
import cv2
import os #hy: for convert avi
import imutils
import numpy as np

settings.set_global()

#def crop_resize_gray(frame, )


#FILE = '/home/hamed/CCTV/WheelChair_Detect/Test_Videos/2015-12-10_13.51_58.0.cam_81_1.mp4'
#FILE = './Test_Videos/2015-12-10_13.09_32.0.cam_81_1.mp4'
#FILE = './Test_Videos/2015-12-10_10.49_21.0.cam_80_1.mp4'



FILE = './Test_Videos/hinten.mp4'

#while True:

video = cv2.VideoCapture(FILE)
video.set(1,2) #hy: added
w_window = 200 #hy: window width for displaying image


while True:
#while(video.isOpened()):
    ret, frame = video.read()
    if ret: #hy: ensure to execute imshow only when there is a frame has been caught
        #cv2.rectangle(frame, (610, 350), (1300, 750), color=155, thickness=4)
        #cv2.rectangle(frame, (settings.crop_x1, settings.crop_y1), (settings.crop_x2, settings.crop_y2), color=155, thickness=4)
        #frame_crop = frame[350:750, 610:1300] #values for wheelchair test video hy: [y1:y2,x1:x2]
        #frame_crop = frame[0:780, 0:1280]
        #frame_crop = frame[settings.crop_y1:settings.crop_y2, settings.crop_x1:settings.crop_x2]
        #frame_crop_resize_gray = imutils.resize(cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY), width=42)
        #frame_crop_resize_gray = imutils.resize(cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY), width=w_window) #hy: width=width of window


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)


            #cv2.imshow("Full frame", frame)
            #cv2.imshow("Video_Cropped_Resized", frame_crop_resize_gray)


        if cv2.waitKey(1) & 0xFF == ord('q'):

            break
            #video.release()
            #cv2.destroyAllWindows()

    else:
        print('No further frame to read.')
        break



video.release()

cv2.destroyAllWindows()




