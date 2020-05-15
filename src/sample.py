from subprocess import Popen, PIPE, STDOUT
import tkinter as tk
from tkinter import Tk
from threading import Thread

def create_worker(target):
    return Thread(target=target)


def start_worker(worker):
    worker.start()


def commande():
    cmd = 'ping 8.8.8.8'
    p = Popen(cmd.split(), stdout=PIPE, stderr=STDOUT)
    for line in iter(p.stdout.readline, ''):
        result.configure(text=line)

root = Tk()
root.geometry('600x80+400+400')

worker = create_worker(commande)
tk.Button(root, text='Ping', command=lambda: start_worker(worker)).pack()

result = tk.Label(root)
result.pack()

root.mainloop()