import numpy as np
import tensorflow.keras as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import * 
from tensorflow.keras.preprocessing import image
import pandas as pd
import click
from PIL import Image
from time import sleep
#RSA
# STEP 1: Generate Two Large Prime Numbers (p,q) randomly
from random import randrange, getrandbits
from tkinter import *
from tkinter import ttk  
from tkinter import Menu  
from tkinter import messagebox as mbox  
# import filedialog module
from tkinter import filedialog
flg=0;
import tkinter as tk

# Function for opening the
# file explorer window
def browseFiles():
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a CSV File",
                                          filetypes = (("CSV files",
                                                        "*.csv*"),
                                                       ("all files",
                                                        "*.*")))
    # Change label contents
    label_file_explorer.configure(text="File Opened: "+filename)
    global f
    f = filename

##
def start():

    print("Process Started")
    dataset = pd.read_csv(f)
    dataset=dataset.dropna(how="any")
    print(dataset)

    print(dataset.info())

    X = dataset.iloc[:,:13].values

    model = tf.models.load_model("lstm_train.h5")

    ypred = model.predict(X)
    ypred = ypred.round()
    print(ypred)
    app = tk.Tk()
    ##
    if(ypred[0,0]==0):
        print("Cardiovascular disease status : Not Detected")
        label_file_explorer.configure(text="For the given dataset the Predicted Value is Absence of Cardiovascular disease")
        app.title("Prediction of Cardiovascular disease ")
        ttk.Label(app, text="Result for the request:Cardiovascular disease Detected").grid(column=0,row=0,padx=20,pady=30)  
        menuBar = Menu(app)
        app.config(menu=menuBar)
    else:
        print("Cardiovascular disease status : Detected")
        label_file_explorer.configure(text="For the given dataset the Predicted value is Presence of Cardiovascular disease")
        app.title("Prediction Of Cardiovascular disease ")
        ttk.Label(app, text="Result for the request:Cardiovascular disease Not Detected").grid(column=0,row=0,padx=20,pady=30)
        menuBar = Menu(app)
        app.config(menu=menuBar)
    
 ##       
if __name__ == '__main__':
    window = Tk()
  
    # Set window title
    window.title('Prediction Of Cardiovascular disease')
      
    # Set window size
    window.geometry("700x400")
      
    #Set window background color
    window.config(background = "white")
      
    # Create a File Explorer label
    label_file_explorer = Label(window,
                                text = "Please give Input Request",
                                width = 100, height = 4,
                                fg = "blue")
         
    button_explore = Button(window,
                            text = "Browse Request Files",
                            command = browseFiles)
    button_exit = Button(window,
                         text = "exit",
                         command = exit)  
    button_start = Button(window,
                         text = "Start Analyzing Request",
                         command = start)

       
    # Grid method is chosen for placing
    # the widgets at respective positions
    # in a table like structure by
    # specifying rows and columns
    label_file_explorer.grid(column = 1, row = 1, padx=5, pady=5)
    button_explore.grid(column = 1, row = 3, padx=5, pady=5)
    button_exit.grid(column = 1,row = 9, padx=5, pady=5)
    button_start.grid(column = 1,row = 12, padx=5, pady=5)
      
    # Let the window wait for any events
    
    
    window.mainloop()


    
