import tkinter as tk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox, ttk
import threading
import queue
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mask_maker_functions import phase_mask_creation, phase_mask_creation_folder, make_the_gaussian, create_trans_mask



class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.progress_bar = ttk.Progressbar(self, length=400)
        self.progress_bar.pack()

        self.status_label = tk.Label(self, text="Idle")
        self.status_label.pack()

progress_queue = queue.Queue()


def select_input():
    path = filedialog.askdirectory()
    input_entry_folder.delete(0, tk.END)
    input_entry_folder.insert(0, path)

def select_input_comp():
    path = filedialog.askopenfilename()
    input_entry_comp.delete(0, tk.END)
    input_entry_comp.insert(0, path)

def select_output():
    path = filedialog.askdirectory()
    output_entry.delete(0, tk.END)
    output_entry.insert(0, path)

def display_figure(fig):

    # Remove old plot if it exists
    for widget in trans_frame.winfo_children():
        if isinstance(widget, FigureCanvasTkAgg):
            widget.get_tk_widget().destroy()

    canvas = FigureCanvasTkAgg(fig, master=trans_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=8, column=1, columnspan=3)

def run():
    mode = mode_var.get()

    try:
        if mode == "phase":
            input_dir = input_entry_folder.get()
            input_comp = input_entry_comp.get()
            output_dir = output_entry.get()

            if not input_dir or not output_dir:
                messagebox.showerror("Error", "Please select required folders")
                return

            phase_mask_creation_folder(input_dir, input_comp, output_dir, plot=False)

        elif mode == "trans":
        
            try:
                lambda_c = int(center_entry.get())
                slm_size = int(slm_entry.get())
                bandwidth = int(sigma_entry.get())
                output_dir = output_entry.get()
                

            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers.")
                return
            condition = display_var.get()
           
            fig = make_the_gaussian(lambda_c,slm_size, bandwidth, output_dir, Display = condition)
            if condition and fig is not None:
                display_figure(fig)

        messagebox.showinfo("Success", "Processing completed!")

    except Exception as e:
        messagebox.showerror("Error", str(e))    
'''
def run():
    input_dir = input_entry_folder.get()
    input_comp = input_entry_comp.get()
    output_dir = output_entry.get()

    if not input_dir or not output_dir:
        messagebox.showerror("Error", "Please select both folders and the Compensation Mask")
        return
    
    try:
        phase_mask_creation_folder(input_dir,input_comp, output_dir, plot = False)
        messagebox.showinfo("Success", "Processing completed!")
    except Exception as e:
        messagebox.showerror("Error", str(e))
'''    
def update_ui():
    if mode_var.get() == "phase":
        trans_frame.grid_remove()
        phase_frame.grid()
    else:  # trans mode
        phase_frame.grid_remove()
        trans_frame.grid()

root = tk.Tk()


root.geometry('800x400')
root.title("MIIPS Mask Creation Tool")
mode_var = tk.StringVar(value="phase") 
tk.Radiobutton(root, text="Phase Mask",
               variable=mode_var,
               value="phase",
               command=lambda: update_ui(),
               bg="green", fg="white").grid(row=0, column=1)

tk.Radiobutton(root, text="Trans Mask",
               variable=mode_var,
               value="trans",
               command=lambda: update_ui(),
               bg="green", fg="white").grid(row=0, column=2)

phase_frame = tk.Frame(root)
phase_frame.grid(row=1, column=0, columnspan=3)



tk.Label(phase_frame, text="Input Folder").grid(row=1, column=0)
input_entry_folder = tk.Entry(phase_frame, width=50)
input_entry_folder.grid(row=1, column=1)
tk.Button(phase_frame, text="Browse", command=select_input).grid(row=1, column=2)

tk.Label(phase_frame, text="Compensation Mask").grid(row=2, column=0)
input_entry_comp = tk.Entry(phase_frame, width=50)
input_entry_comp.grid(row=2, column=1)
tk.Button(phase_frame, text="Browse", command=select_input_comp).grid(row=2, column=2)

tk.Label(phase_frame, text="Save Folder").grid(row=3, column=0)
output_entry = tk.Entry(phase_frame, width=50)
output_entry.grid(row=3, column=1)
tk.Button(phase_frame, text="Browse", command=select_output).grid(row=3, column=2)

tk.Button(phase_frame, text="Create Masks", command=run, bg="green", fg="black").grid(row=4, column=1)

trans_frame = tk.Frame(root)
trans_frame.grid(row=1, column=0, columnspan=3)

display_var = tk.BooleanVar(value=False)

tk.Label(trans_frame, text="Display Result?").grid(row=5, column=0)

tk.Radiobutton(trans_frame,
               text="Yes",
               variable=display_var,
               value=True).grid(row=5, column=1)

tk.Radiobutton(trans_frame,
               text="No",
               variable=display_var,
               value=False).grid(row=5, column=2)


tk.Label(trans_frame, text="SLM Size (pixels)").grid(row=1, column=0)
slm_entry = tk.Entry(trans_frame, width=20)
slm_entry.grid(row=1, column=1)

tk.Label(trans_frame, text="Bandwidth (nm)").grid(row=2, column=0)
sigma_entry = tk.Entry(trans_frame, width=20)
sigma_entry.grid(row=2, column=1)

tk.Label(trans_frame, text="Central Wavelength (nm)").grid(row=3, column=0)
center_entry = tk.Entry(trans_frame, width=20)
center_entry.grid(row=3, column=1)

tk.Label(trans_frame, text="Save Folder").grid(row=4, column=0)
output_entry = tk.Entry(trans_frame, width=20)
output_entry.grid(row=4, column=1)
tk.Button(trans_frame, text="Browse", command=select_output).grid(row=4, column=2)

tk.Button(trans_frame, text="Create Gaussian Mask",
          command=run,
          bg="green", fg="black").grid(row=6, column=1)

update_ui()
root.mainloop()

