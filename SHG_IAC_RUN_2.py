# gui.py
import tkinter as tk
import matplotlib

import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox, ttk
import threading
import queue
import scipy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from SHG_IAC_functions import FROG_trace_folder

'''
class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.progress_bar = ttk.Progressbar(self, length=400)
        self.progress_bar.pack()

        self.status_label = tk.Label(self, text="Idle")
        self.status_label.pack()
'''
progress_queue = queue.Queue()

import sys
import os

#def resource_path(relative_path):
 #   try:
  #      base_path = sys._MEIPASS
   # except Exception:
    #    base_path = os.path.abspath(".")
    #return os.path.join(base_path, relative_path)






def select_input_plot():
    path = filedialog.askdirectory()
    input_entry_plot.delete(0, tk.END)
    input_entry_plot.insert(0, path)

def select_output_plot():
    path = filedialog.askdirectory()
    output_entry_plot.delete(0, tk.END)
    output_entry_plot.insert(0, path)


def select_input_no_plot():
    path = filedialog.askdirectory()
    input_entry_no_plot.delete(0, tk.END)
    input_entry_no_plot.insert(0, path)

def select_output_no_plot():
    path = filedialog.askdirectory()
    output_entry_no_plot.delete(0, tk.END)
    output_entry_no_plot.insert(0, path)


def display_figure(fig):

    # Remove old plot if it exists
    for widget in plot_frame.winfo_children():
        if isinstance(widget, FigureCanvasTkAgg):
            widget.get_tk_widget().destroy()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=8, column=1, columnspan=3)

def run():

    mode = mode_var.get()

    try:
        if mode =="plot":

            input_dir = input_entry_plot.get()
            output_dir = output_entry_plot.get()

            if not input_dir or not output_dir:
                messagebox.showerror("Error", "Please select both folders")
                return
            
            
            fig = FROG_trace_folder(input_dir, output_dir, Relationship= True)
            if fig is not None:
                display_figure(fig)

                


            messagebox.showinfo("Success", "Processing completed!")
          
        elif mode == "no plot":
        
            input_dir = input_entry_no_plot.get()
            output_dir = output_entry_no_plot.get()

            if not input_dir or not output_dir:
                messagebox.showerror("Error", "Please select both folders")
                return
            
            
            FROG_trace_folder(input_dir, output_dir, Relationship= False)
            messagebox.showinfo("Success", "Processing completed!")
            

        

    except Exception as e:
        messagebox.showerror("Error", str(e))    



def update_ui():
    if mode_var.get() == "no plot":
        noplot_frame.grid_remove()
        noplot_frame.grid()
    #else:  # trans mode
     #   plot_frame.grid_remove()
      #  noplot_frame.grid()
   
    
root = tk.Tk()
root.geometry('800x400')
root.title("IAC Temporal Width Recoverer")
#icon_path = resource_path("SHG_IAC_icon.png")
#icon = tk.PhotoImage(file=icon_path)
#root.iconphoto(True, icon)
mode_var = tk.StringVar(value="plot") 
'''
tk.Radiobutton(root, text="Plot Relationship",
               variable=mode_var,
               value="plot",
               command=lambda: update_ui(),
               bg="green", fg="black").grid(row=0, column=1)
'''
tk.Radiobutton(root, text="Don't Plot Relationship",
               variable=mode_var,
               value="no plot",
               command=lambda: update_ui(),
               bg="green", fg="black").grid(row=0, column=2)
                                                                                                               
               
'''
plot_frame = tk.Frame(root)
plot_frame.grid(row=1, column =0 , columnspan =3)


tk.Label(plot_frame, text="Input Folder").grid(row=2, column=0)
input_entry_plot = tk.Entry(plot_frame, width=50)
input_entry_plot.grid(row=2, column=1)
tk.Button(plot_frame, text="Browse", command=select_input_plot).grid(row=2, column=2)

tk.Label(plot_frame, text="Save Folder").grid(row=3, column=0)
output_entry_plot = tk.Entry(plot_frame, width=50)
output_entry_plot.grid(row=3, column=1)
tk.Button(plot_frame, text="Browse", command=select_output_plot).grid(row=3, column=2)

tk.Button(plot_frame, text="Recover", command=run, bg="green", fg="black").grid(row=4, column=1)
'''
noplot_frame = tk.Frame(root)
noplot_frame.grid(row=1, column =0 , columnspan =3)


tk.Label(noplot_frame, text="Input Folder").grid(row=2, column=0)
input_entry_no_plot = tk.Entry(noplot_frame, width=50)
input_entry_no_plot.grid(row=2, column=1)
tk.Button(noplot_frame, text="Browse", command=select_input_no_plot).grid(row=2, column=2)

tk.Label(noplot_frame, text="Save Folder").grid(row=3, column=0)
output_entry_no_plot = tk.Entry(noplot_frame, width=50)
output_entry_no_plot.grid(row=3, column=1)
tk.Button(noplot_frame, text="Browse", command=select_output_no_plot).grid(row=3, column=2)

tk.Button(noplot_frame, text="Recover", command=run, bg="green", fg="black").grid(row=4, column=1)














update_ui()
root.mainloop()




#try:
 #       FROG_trace_folder(input_dir, output_dir, Relationship= True)
  #      messagebox.showinfo("Success", "Processing completed!")
   # except Exception as e:
    #    messagebox.showerror("Error", str(e))
