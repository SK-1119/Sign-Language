#==============================================================
#  Author: Kunal SK Sukhija
#==============================================================

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import joblib
from tkinter import *
f=Tk()
f.geometry("500x500")
f.title("Sign Language Recognition System")
f.configure(bg="#1F575C")
l1=Label(f,text="Sign Language Recognition System",font=('Comic Sans MS',20,'italic','bold','underline'))
l1.place(x=23.5,y=30)
l2=Label(f,text=f"Predicted Letter is:  \n Confidence is: \n",font=('Comic Sans MS',18,'italic','bold',))
l2.place(x=40,y=160)
f.mainloop()