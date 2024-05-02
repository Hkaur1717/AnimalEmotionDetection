import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model # type: ignore

# Load the trained model for emotion prediction
# Replace 'model_path' with the path to your trained model file
model_path = 'path/to/your/trained_model.h5'
model = load_model(model_path)

# Define the class labels (e.g., happy, sad, hungry)
emotion_classes = {0: 'happy', 1: 'sad', 2: 'hungry'}

# Define a function to load and preprocess the image

def load_and_preprocess_image(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    
    # Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to the input shape expected by the model (e.g., 48x48)
    img = cv2.resize(img, (48, 48))
    
    # Normalize the image (0-1 range)
    img = img.astype('float32') / 255.0
    
    # Add the batch and channel dimensions
    img = np.expand_dims(img, axis=(0, -1))
    
    return img

# Define a function to predict emotions and display the result
def predict_emotions():
    # Get the file path of the image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    
    if not file_path:
        # User cancelled the file selection
        return
    
    # Load and preprocess the image
    img = load_and_preprocess_image(file_path)
    
    # Make predictions using the model
    predictions = model.predict(img)
    emotion_probs = predictions[0]
    
    # Find the predicted emotion
    emotion_label = np.argmax(emotion_probs)
    predicted_emotion = emotion_classes.get(emotion_label, "unknown")
    
    # Display the image in the GUI
    img_pil = Image.open(file_path)
    img_tk = ImageTk.PhotoImage(img_pil)
    image_label.config(image=img_tk)
    image_label.image = img_tk
    
    # Trigger a notification with the predicted emotion
    messagebox.showinfo("Emotion Prediction", f"Predicted Emotion: {predicted_emotion}")

# Create the main application window
root = tk.Tk()
root.title("Animal Emotion Prediction")

# Create a button to predict emotions
predict_button = tk.Button(root, text="CAT", command=predict_emotions)
predict_button.pack()

# Create a label to display the loaded image
image_label = tk.Label(root)
image_label.pack()

# Start the main loop of the application
root.mainloop()
