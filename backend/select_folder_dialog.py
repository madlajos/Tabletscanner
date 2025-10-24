# select_folder_dialog.py
import tkinter as tk
from tkinter import filedialog
import sys
import json

root = tk.Tk()
root.withdraw()
folder = filedialog.askdirectory(title="Képmentés mappája")
root.destroy()

# Return the result to stdout
print(json.dumps({"folder": folder}))
sys.exit(0)
