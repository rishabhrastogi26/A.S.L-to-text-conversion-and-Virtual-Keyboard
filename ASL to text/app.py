from flask import Flask,render_template
import numpy as np
import cv2
import wx
import cv2
from cvzone.HandTrackingModule import HandDetector
import math
from cvzone.ClassificationModule import Classifier
import time
import winsound

app = Flask(__name__)
def virtual_keyboard():    
    xs={}
    prev="He"
    lower_b = np.array([100,120,100])
    upper_b = np.array([150,255,255])
    keyboard = np.zeros((450,750,3),np.uint8)
    keys_set3={0:"Q",1:"W",2:"E",3:"R",4:"T",
                5:"Y",6:"U",7:"I",8:"O",9:"P",
                10:"A",11:"S",12:"D",13:"F",14:"G",
                15:"H",16:"J",17:"K",18:"L",19:"' '",
                20:"Z",21:"X",22:"C",23:"V",24:"",
                25:"B",26:"N",27:"M",28:",",29:"cl"
            }
    app =wx.App(False)
    (sx,sy) =wx.DisplaySize()
    kernelOpen = np.ones((4,4))
    kernelClose = np.ones((18,18))
    def letter(letter_index,text):
        x=(letter_index%10)*75
        y=int(letter_index/10)*75
        xs[x,y]=text
        height,width=75,75
        th= 3 
        cv2.rectangle(img, (x+th,y+th), (x+width-th,y+height-th),(100,255,255),th)
        font_scale=4
        font_th =3
        font_letter =  cv2.FONT_HERSHEY_PLAIN
        text_size =cv2.getTextSize(text,font_letter,font_scale,font_th)[0]
        width_text, height_text = text_size[0],text_size[1]
        text_x= int((width-width_text)/2) +x
        text_y = int((height+height_text)/2) +y
        cv2.putText(img,text,(text_x,text_y),font_letter,font_scale,(100,255,255),font_th)
    cam = cv2.VideoCapture(0)
    frame_count=0
    pos=0
    while True:
        _,img = cam.read()
        img= cv2.resize(img,(800,600))
        img = cv2.flip( img, 1)
        for i in range(30):
            letter(i,keys_set3[i])
        imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(imgHsv,lower_b,upper_b)
        maskOpen =cv2.morphologyEx(mask, cv2.MORPH_OPEN,kernelOpen)
        maskClose =cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE,kernelClose)
        conts,h = cv2.findContours(maskClose.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if(len(conts)==1):
            cv2.drawContours(img,conts,-1,(255,0,0),3)
            x1,y1,w1,h1 = cv2.boundingRect(conts[0])
            cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
            height,width=75,75
            th= 3 
            if int(x1/width) <=9 and int(y1/height) <=2:
                curr=keys_set3[int(x1/width)+int(y1/height)*10]
                cv2.rectangle(img, (int(x1/width)*75+th,int(y1/height)*75+th), (int(x1/width)*75+width-th,int(y1/height)*75+height-th),(0,0,255),th)
                if frame_count ==12:
                    cv2.rectangle(img, (int(x1/width)*75+th,int(y1/height)*75+th), (int(x1/width)*75+width-th,int(y1/height)*75+height-th),(-1),th)
                    frame_count=0
                    if curr=='cl':
                        keyboard = np.zeros((450,750,3),np.uint8)
                        pos=0
                    elif curr=='I':
                        cv2.putText(keyboard,curr, (pos,100),  cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),3, 2)
                        pos+=20
                    else:
                        cv2.putText(keyboard,curr, (pos,100),  cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),3, 2)
                        pos+=30
                if(curr!=prev):
                    frame_count=0
                    prev=curr
                else:
                    frame_count+=1
            else:
                prev="He"
        cv2.imshow('virtual', img)
        cv2.imshow('board',keyboard)
        key=cv2.waitKey(1)
        if key == 27:
            break
    cam.release()
    cv2.destroyAllWindows()


def asl_():
    offset = 20
    imgSize = 300
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', '  ', 'thanks', 'hello']
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier("C:/Users/91911/Desktop/Minor2_Final/sid_model/keras_model.h5", "C:/Users/91911/Desktop/Minor2_Final/sid_model/labels.txt")

    file = open("C:/Users/91911/Desktop/Minor2_Final/recognized_characters.txt", "w")
    start_time = time.time()

    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
            imgCropShape = imgCrop.shape
            aspectRatio = h/w

            if aspectRatio > 1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap:wGap+wCal] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                cv2.putText(imgOutput, labels[index], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                k = imgSize/w
                hCal = math.ceil(h*k)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hGap+hCal, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                cv2.putText(imgOutput, labels[index], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if time.time() - start_time > 5:
                print("Predicted character:", labels[index])

                file.write(labels[index])  # Write recognized character to file
                file.flush()  
                start_time = time.time()
                winsound.PlaySound("SystemExit", winsound.SND_ALIAS)  # Play sound when character is written

        cv2.imshow("Image", imgOutput)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    file.close()  # Close file object to save and close the text file
    cv2.destroyAllWindows()
    cap.release()
    return render_template('index.html')

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/vkey',methods=['Get','Post'])
def vkey():
    virtual_keyboard()
    return render_template('index.html')

@app.route('/asl',methods=['Get','Post'])
def asl():
    asl_()
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
