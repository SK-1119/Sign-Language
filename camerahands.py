#==============================================================
#  Author: Kunal SK Sukhija
#==============================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import pandas as pd
from tkinter import *

def change_val():
    J=1
    print(f"value changed J={J}")
    for key,value in letter_prediction_dict.items():
        if value==high2 and J==1:
            print("Predicted Character 2: ", key)
            print('Accuracy 2: ', 100*value)
            l3=Label(f,text=f"The Character may be:  {key}\n Accuracy: {100*value}%",font=('Comic Sans MS',18,'italic','bold',))
            l3.place(x=25,y=315)
        elif value==high3 and J==1:
            print("Predicted Character 3: ", key)
            print('Accuracy 3: ', 100*value)
            l4=Label(f,text=f"The Character may be:  {key}\n Accuracy: {100*value}%",font=('Comic Sans MS',18,'italic','bold',))
            l4.place(x=22,y=400)
        elif value==high4 and J==1:
            print("Predicted Character 4: ", key)
            print('Accuracy 4: ', 100*value)
            l5=Label(f,text=f"The Character may be:  {key}\n Accuracy: {100*value}%",font=('Comic Sans MS',18,'italic','bold',))
            l5.place(x=22,y=485)
    b3=Button(f,text="OK",font=('Comic Sans MS',18,'italic','bold',),command=lambda:f.destroy())
    b3.place(x=225,y=575)

model = load_model('smnist.h5')
J=0
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

_, frame = cap.read()

h, w, c = frame.shape

img_counter = 0
analysisframe = ''
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
while True:
    _, frame = cap.read()

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        analysisframe = frame
        showframe = analysisframe
        cv2.imshow("Frame", showframe)
        framergbanalysis = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2RGB)
        resultanalysis = hands.process(framergbanalysis)
        hand_landmarksanalysis = resultanalysis.multi_hand_landmarks
        if hand_landmarksanalysis:
            for handLMsanalysis in hand_landmarksanalysis:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lmanalysis in handLMsanalysis.landmark:
                    x, y = int(lmanalysis.x * w), int(lmanalysis.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                y_min -= 20
                y_max += 20
                x_min -= 20
                x_max += 20 

        analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
        analysisframe = analysisframe[y_min:y_max, x_min:x_max]
        analysisframe = cv2.resize(analysisframe,(28,28))


        nlist = []
        rows,cols = analysisframe.shape
        for i in range(rows):
            for j in range(cols):
                k = analysisframe[i,j]
                nlist.append(k)
        
        datan = pd.DataFrame(nlist).T
        colname = []
        for val in range(784):
            colname.append(val)
        datan.columns = colname

        pixeldata = datan.values
        pixeldata = pixeldata / 255
        pixeldata = pixeldata.reshape(-1,28,28,1)
        prediction = model.predict(pixeldata)
        predarray = np.array(prediction[0])
        letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
        predarrayordered = sorted(predarray, reverse=True)
        high1 = predarrayordered[0]
        high2 = predarrayordered[1]
        high3 = predarrayordered[2]
        high4 = predarrayordered[3]
        
        
        
        f=Tk()
        f.geometry("500x700")
        f.title("Sign Language Recognition System")
        f.configure(bg="#1F575C")
        l1=Label(f,text="Sign Language Recognition System",font=('Comic Sans MS',20,'italic','bold','underline'))
        l1.place(x=23.5,y=30)
        
        for key,value in letter_prediction_dict.items():
            if value==high1:
                print("Predicted Character 1: ", key)
                print('Accuracy 1: ', 100*value)
                l2=Label(f,text=f"Predicted Character is:  {key}\n Accuracy: {100*value}%",font=('Comic Sans MS',18,'italic','bold',))
                l2.place(x=35,y=100)
                l5=Label(f,text="Want to see 3 other Possiblities?",font=('Comic Sans MS',18,'italic','bold',))
                l5.place(x=50,y=185)
                b1=Button(f,text="Yes",font=('Comic Sans MS',18,'italic','bold',),command=lambda:change_val())
                b1.place(x=120,y=240)
                b2=Button(f,text="No",font=('Comic Sans MS',18,'italic','bold',),command=lambda:f.destroy())
                b2.place(x=320,y=240)
        f.mainloop()

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
cap.release()
cv2.destroyAllWindows()
