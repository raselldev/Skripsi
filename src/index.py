from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from Model import Model, DecoderType
from main import infer
import sub

root = Tk()
root.title("Handwriting Detection")

mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0,)
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

fnCharList = '../model/charList.txt'
fnAccuracy = '../model/accuracy.txt'
fnTrain = '../data/'
#fnInfer = '../data/test.png'
#fnInfer = 
fnCorpus = '../data/corpus.txt'
decoderType = DecoderType.BestPath


def upload():
    root.filename =  filedialog.askopenfilename(initialdir = "/Users/Documents",title = "Select file",filetypes = (("jpeg files","*.png"),("all files","*.*")))
    return

def detect():
    root.filename =  filedialog.askopenfilename(initialdir = "/Users/Documents",title = "Select file",filetypes = (("jpeg files","*.png"),("all files","*.*")))
    print(open(fnAccuracy).read())
    model = Model(open(fnCharList).read(), decoderType, mustRestore=True)
    infer(model, root.filename)


ttk.Button(mainframe, text="Pilih Gambar", command=upload).grid(column=1, row=1)
ttk.Button(mainframe, text="Detect", command=detect).grid(column=1, row=2)
#ttk.Label(mainframe, textvariable=uploadText).grid(column=2, row=1)


#ttk.Button(mainframe, text="Ok").grid(column=2, row=1)
#ttk.Label(mainframe, text="ASD").grid(column=2, row=1)
#ttk.Label(mainframe, text="ASD").grid(column=3, row=1)

for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)


root.mainloop()