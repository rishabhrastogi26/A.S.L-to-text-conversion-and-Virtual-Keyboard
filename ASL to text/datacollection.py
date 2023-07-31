#datacollection and detect
#300*300 will be fixed size of square, if image small, will be placed at center.
#cvzone, mediapipe,tensorflow libreries are installed. 
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np#to create white image's matrix
import math
import time#to put variable names to images

offset=20#to increase image size
imgSize=300
counter =0
folder="C:/Users/91911/Desktop/Final/Dataset/manu"
cap=cv2.VideoCapture(0)#0 is Id no of webcam.
detector=HandDetector(maxHands=1)
while True:
    success, img=cap.read()
    hands, img=detector.findHands(img)

    if hands:
        hand=hands[0]#we have only 1 hand
        x,y,w,h=hand['bbox']#hand is a dictionary, bbox=bounding_box
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        '''The line of code imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255 creates a NumPy array with dimensions (imgSize, imgSize, 3)
        and data type np.uint8.
        Each element of the array is initialized to the value 1.0 because of the np.ones function. The third dimension of the array is 3,
        representing the three color channels (red, green, and blue) used in an RGB image.
        The multiplication by 255 changes the values of the elements to 255, effectively setting each pixel to be white,
        since the RGB values for white are (255, 255, 255).
        Therefore, the resulting array imgWhite represents a white image with dimensions (imgSize, imgSize).'''
        imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]

        imgCropShape=imgCrop.shape
        aspectRatio=h/w#height/width

        if aspectRatio>1:#that means height>width
            k=imgSize/h#here 300/h
            wCal=math.ceil(k*w)
            imgResize=cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape=imgResize.shape
            wGap=math.ceil((imgSize-wCal)/2)
            imgWhite[:,wGap:wGap+wCal]=imgResize#putting image at center of white image.


        else:
            k=imgSize/w
            hCal=math.ceil(h*k)
            imgResize=cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape=imgResize.shape
            hGap=math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hGap+hCal,:]=imgResize

        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)


    cv2.imshow("Image",img)
    key=cv2.waitKey(1)#1 ms delay
    if key==ord("s"):#starts collecting when we press s
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)


##Trained model and downloaded it from website teachingmachine.

