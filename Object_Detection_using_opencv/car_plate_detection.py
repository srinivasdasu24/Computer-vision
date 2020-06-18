
import cv2
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

img = cv2.imread('Images/car_plate.jpg')


def display(img):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    new_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    ax.imshow(new_img)
    

display(img)


plate_classifier = cv2.CascadeClassifier("Haarcascades/haarcascade_plate_number.xml")
def detect_plate(img):
    plate_img = img.copy()
    plate_rects = plate_classifier.detectMultiScale(plate_img,1.1,1)
    
    for (x,y,w,h) in plate_rects:
        cv2.rectangle(plate_img,
                     (x,y),
                     (x+w,y+h),
                     (255,255,255),
                     8)
    return plate_img
    
 plt.imshow(detect_plate(img))
