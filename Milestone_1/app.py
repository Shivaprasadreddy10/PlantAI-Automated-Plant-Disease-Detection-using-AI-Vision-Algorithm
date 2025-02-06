from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
model = load_model('plant_disease_model.keras')

# Define allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return 'No file uploaded!'

    # Check if the file is valid
    if not allowed_file(file.filename):
        return 'Invalid file format! Please upload an image file (PNG, JPG, JPEG, or GIF).'
    
    img = Image.open(file)
    img = preprocess_image(img)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]

    return f'Predicted Class: {predicted_class}'

if __name__ == "__main__":
    app.run(debug=True)