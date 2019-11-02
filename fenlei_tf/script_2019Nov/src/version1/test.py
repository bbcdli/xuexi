import numpy as np
import imutils
import cv2

# Load an color image in grayscale
#img = cv2.imread('/home/hamed/WheelChair_Detect_copy/Data/Kiwa/2015-12-10_10.49_21.1.cam_81_1_020220.jpg')
img = cv2.imread('/home/hamed/Documents/Lego_copy/Data/hinten/20160421_133435_croppad_227_1_1.jpg')

cv2.imshow("test",img)

rotated=imutils.rotate(img,angle=5)

cv2.imshow("Rotated",rotated)

gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Gray",gray)

grayEq=gray

cv2.equalizeHist(gray, grayEq)

cv2.imshow("GrayEq", grayEq)

gray=cv2.flip(grayEq,-1)

cv2.imshow("flipped",gray)

cv2.imshow("reesize",imutils.resize(imutils.resize(gray,width=84),width=210)) #hy:updated size as per the example image old value 84,210

cv2.waitKey(0)
cv2.destroyAllWindows()

