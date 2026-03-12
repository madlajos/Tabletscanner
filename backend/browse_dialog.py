# browse_dialog.py — Subprocess-based file/folder browser dialogs.
# Invoked by Flask endpoints to avoid Tkinter main-thread issues.
# Usage:
#   python browse_dialog.py file    — opens image file selection dialog
#   python browse_dialog.py values  — opens CSV/TXT values file selection dialog
#   python browse_dialog.py folder  — opens folder selection dialog
import tkinter as tk
from tkinter import filedialog
import sys
import json

mode = sys.argv[1] if len(sys.argv) > 1 else "folder"

root = tk.Tk()
root.withdraw()
# Force the dialog to appear in the foreground
root.attributes('-topmost', True)
root.update()

if mode == "file":
    path = filedialog.askopenfilename(
        parent=root,
        title="Képfájl kiválasztása",
        filetypes=[("Képfájlok", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("Minden fájl", "*.*")]
    )
elif mode == "values":
    path = filedialog.askopenfilename(
        parent=root,
        title="Értékfájl kiválasztása",
        filetypes=[("Értékfájlok", "*.csv *.txt"), ("CSV", "*.csv"), ("Szövegfájl", "*.txt")]
    )
else:
    path = filedialog.askdirectory(parent=root, title="Képek mappája")

root.destroy()

print(json.dumps({"path": path or ""}))
sys.exit(0)
