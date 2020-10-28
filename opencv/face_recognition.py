import cv2
import matplotlib.pyplot as plt
import numpy as np

def show(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
def imread(image):
   
    image = cv2.imread(image)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image
def facedetect(image):
    image = imread(image)
    # 级联分类器
    detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=2, minSize=(10,10), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in rects:
        # 画矩形框
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
    show(image)
facedetect('./imgs/face.png')
