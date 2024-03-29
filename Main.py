import tkinter as tk
from tkinter import Message ,Text
from PIL import Image, ImageTk
import pandas as pd

import tkinter.ttk as ttk
import tkinter.font as font
import tkinter.messagebox as tm
import matplotlib.pyplot as plt
import csv
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter.messagebox as tm
import RandomForest as RF
import SVM as SV
import DecisionTree as DT
from sklearn.externals import joblib
import Single as sg

bgcolor="#DAF7A6"
bgcolor1="#B7C526"
fgcolor="black"


def Home():
	global window
	def clear():
	    print("Clear1")
	    txt.delete(0, 'end')
	    txt1.delete(0, 'end')



	window = tk.Tk()
	window.title("Detecting Phising Website")

 
	window.geometry('1280x720')
	window.configure(background=bgcolor)
	#window.attributes('-fullscreen', True)

	window.grid_rowconfigure(0, weight=1)
	window.grid_columnconfigure(0, weight=1)
	

	message1 = tk.Label(window, text="Detecting Phising Website" ,bg=bgcolor  ,fg=fgcolor  ,width=50  ,height=3,font=('times', 30, 'italic bold underline')) 
	message1.place(x=100, y=20)

	lbl = tk.Label(window, text="Select Dataset",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
	lbl.place(x=100, y=200)
	
	txt = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
	txt.place(x=400, y=215)

	lbl1 = tk.Label(window, text="Enter Url",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
	lbl1.place(x=100, y=300)
	
	txt1 = tk.Entry(window,width=40,bg="white" ,fg="black",font=('times', 15, ' bold '))
	txt1.place(x=400, y=315)

	def browse():
		path=filedialog.askopenfilename()
		print(path)
		txt.insert('end',path)
		if path !="":
			print(path)
		else:
			tm.showinfo("Input error", "Select Train Dataset")	


	def RFprocess():
		sym=txt.get()
		if sym != "" :
			RF.process(sym)
			tm.showinfo("Input", "RANDOM FOREST Successfully Finished")
		else:
			tm.showinfo("Input error", "Select Dataset")


	def SVMprocess():
		sym=txt.get()
		if sym != "" :
			SV.process(sym)
			tm.showinfo("Input", "SVM Successfully Finished")
		else:
			tm.showinfo("Input error", "Select Dataset")

	def DTprocess():
		sym=txt.get()
		if sym != "" :
			DT.process(sym)
			tm.showinfo("Input", "DecisionTree Successfully Finished")
		else:
			tm.showinfo("Input error", "Select Dataset")

	def Predicted():
		sym=txt.get()
		sym1=txt1.get()
		if sym != "" and sym1!="" :
			prediction=sg.process(sym,sym1)
			tm.showinfo("Input", "Predicted Class is  "+ str(prediction[0]))
		else:
			tm.showinfo("Input error", "Select Dataset and Enter Url")

	browse = tk.Button(window, text="Browse", command=browse  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	browse.place(x=650, y=200)

	clearButton = tk.Button(window, text="Clear", command=clear  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
	clearButton.place(x=950, y=200)

	

	RFbutton = tk.Button(window, text="RANDOM FOREST", command=RFprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=16  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	RFbutton.place(x=230, y=600)


	SVMbutton = tk.Button(window, text="SVM", command=SVMprocess  ,fg=fgcolor   ,bg=bgcolor1 ,width=13  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	SVMbutton.place(x=450, y=600)


	DTbutton = tk.Button(window, text="DecisionTree", command=DTprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	DTbutton.place(x=650, y=600)

	PRbutton = tk.Button(window, text="Prediction", command=Predicted  ,fg=fgcolor   ,bg=bgcolor1   ,width=16  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	PRbutton.place(x=820, y=600)


	quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg=fgcolor   ,bg=bgcolor1  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	quitWindow.place(x=1100, y=600)

	window.mainloop()
Home()

