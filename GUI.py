import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import easygui
import numpy as np
import tensorflow as tf
import os
from numpy import loadtxt
from pathlib import Path

lines = loadtxt("data/test_classes.txt", delimiter=",", unpack=False)

def open_image():
    # Use easygui's fileopenbox to select an image file
    file_path = easygui.fileopenbox()
    global text_label

    if file_path:
        # Load the selected image
        image = Image.open(file_path)
        image2 = image.resize((32, 32))  # Resize the image to the required dimensions
        image2 = np.array(image2) / 255.0  # Normalize pixel values to the range [0, 1]
        image2 = np.expand_dims(image2, axis=0)  # Add a batch dimension

        # Perform prediction
        prediction = model.predict(image2)
        # Interpret the prediction
        predicted_class = np.argmax(prediction, axis=1)
        print(classes[predicted_class[0]])
        print(classes[predicted_class[0]] == classes[int(lines[int(Path(file_path).stem)])])

        # Resize the image to fit the window if necessary
        max_width = 500
        max_height = 500
        if image.width > max_width or image.height > max_height:
            image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        # Display the image in the Tkinter app
        image_tk = ImageTk.PhotoImage(image)
        label.configure(image=image_tk)
        label.image = image_tk

        if classes[predicted_class[0]] == classes[int(lines[int(Path(file_path).stem)])]:
            text_label.config(text="Odpowiedź: " + classes[predicted_class[0]], fg="green")
        else:
            text_label.config(text="Odpowiedź: " + classes[predicted_class[0]], fg="red")

        text_label2.config(text="Poprawne: " + classes[int(lines[int(Path(file_path).stem)])])

model = tf.keras.models.load_model('WSI.h5')
classes = ["Wielka Niedźwiedzica", "Lutnia", "Łabędź", "Skorpion", "Woźnica", "Strzelec", "Orzeł",
           "Cefeusz", "Bliźnięta", "Mały Pies", "Lew", "Siedem Sióstr", "Byk", "Orion",
           "Wielki Pies", "Księżyc"]


# Create a Tkinter window
window = tk.Tk()
window.title("WSI")

# Create a button to open the image
button = tk.Button(window, text="Open Image", command=open_image)
button.pack()

# Create a label to display the image
label = tk.Label(window)
label.pack()

# Create a label for text at the bottom
text_label2 = tk.Label(window, text="Poprawne:", fg="black", font=("Arial 20 bold"))
text_label2.pack(side=tk.BOTTOM, pady=10)

# Create a label for text at the bottom
text_label = tk.Label(window, text="Odpowiedź:", fg="black", font=("Arial 20 bold"))
text_label.pack(side=tk.BOTTOM, pady=10)

window.eval('tk::PlaceWindow . center')
# Start the Tkinter event loop
window.mainloop()
