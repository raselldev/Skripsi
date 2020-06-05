import tkinter as tk
from tkinter import *
from tkinter import filedialog, Text
from tkinter import RIGHT, BOTH, RAISED
from PIL import ImageTk, Image
from main import *
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
import os

root = tk.Tk()
root.title("Deteksi Tulisan")
apps = []
showDetect = []
showDetect1 = []
decoderType = DecoderType.BestPath

def addFile():
    filename = filedialog.askopenfilename(initialdir="/", title="Select File", filetypes=(("images", "*.png"), ("all files", "*.*")))

    apps.append(filename)
    print(filename)
    for app in apps:
        label = tk.Label(frame, text=app, bg="gray").pack()

    img = Image.open(filename)
    img = img.resize((150, 150), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(frame, image=img)
    panel.image = img
    panel.pack()
    return filename

def resetBtn():
    #for widget in frame.winfo_children():
    #    widget.destroy()
    #frame.destroy()
    #frame = tk.Frame(root, bg="white")
    #frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
    python = sys.executable
    os.execl(python, python, * sys.argv)

def infer(model, fnImg):
    img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)

    showDetect.append(recognized)
    for showdetect in showDetect:
        showD = tk.Label(frame, text=recognized, bg="gray").pack()
    
    showDetect1.append(probability)
    for showdetect1 in showDetect1:
        showD = tk.Label(frame, text=probability, bg="gray").pack()




def runDetect():
    filename = filedialog.askopenfilename(initialdir="/", title="Select File", filetypes=(("images", "*.png"), ("all files", "*.*")))

    apps.append(filename)
    print(filename)
    for app in apps:
        label = tk.Label(frame, text=app, bg="gray").pack()

    img = Image.open(filename)
    img = img.resize((150, 150), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = tk.Label(frame, image=img)
    panel.image = img
    panel.pack()

    print(open(FilePaths.fnAccuracy).read())
    model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True)
    infer(model, filename)
    #recog = tk.Label(frame, text=recognized[0], bg="gray").pack()

canvas = tk.Canvas(root, height=500, width=500, bg="#263D42")
canvas.pack()

frame = tk.Frame(root, bg="white")
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

"""openFile = tk.Button(root, text="Open File", padx=10, pady=5, command=addFile)
openFile.pack(side=RIGHT, padx=5, pady=5)"""

runDetect = tk.Button(root, text="Detect", padx=10, pady=5, command=runDetect)
runDetect.pack(side=RIGHT, padx=5, pady=5)

resetBtn = tk.Button(root, text="Reset", padx=10, pady=5, command=resetBtn)
resetBtn.pack(side=RIGHT, padx=5, pady=5)


root.mainloop() 

if __name__ == '__main__':
    root = tk.Tk()