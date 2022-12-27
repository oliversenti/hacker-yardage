#Import the required libraries
from tkinter import *

#Create an instance of tkinter frame
win= Tk()

#Define the size of window or frame
win.geometry("715x250")

#Set the Menu initially
menu= StringVar()
menu.set("Select Any Language")

#Create a dropdown Menu
drop= OptionMenu(win, menu,"C++", "Java","Python","JavaScript","Rust","GoLang")
drop.pack()

win.mainloop()