from deepface import DeepFace
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt

# Hide main tkinter window
root = tk.Tk()
root.withdraw()

# File selection dialog
file_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
)

if not file_path:
    messagebox.showinfo("No file selected", "You did not select an image.")
    exit()

# Read image
img = cv2.imread(file_path)

# Analyze using DeepFace
try:
    result = DeepFace.analyze(img_path=file_path, actions=['age', 'gender'], enforce_detection=True)
    age = result[0]['age']
    gender = result[0]['gender']

    # Draw result
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f"{gender}, Age: {age}")
    plt.axis('off')
    plt.show()

    messagebox.showinfo("Prediction", f"{gender}, Age: {age}")

except Exception as e:
    messagebox.showerror("Error", f"Could not analyze the image.\n\n{str(e)}")
