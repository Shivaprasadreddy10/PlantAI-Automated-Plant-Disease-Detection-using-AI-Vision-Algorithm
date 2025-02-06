from flask import Flask, render_template, request, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

from disease_info import DISEASE_INFO  # Import the disease information

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
MODEL_PATH = r"C:\Users\nallo\Desktop\Milestone_3\models\PlantAI_Project.keras"
model = load_model(MODEL_PATH)
print("Model Input Shape:", model.input_shape)

# Class mapping
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]


@app.route('/')
def home():
    return render_template('home.html', title='Home', active='home')

@app.route('/detection', methods=['GET', 'POST'])
def detection():
    if request.method == 'POST':
        try:
            uploaded_file = request.files.get('image')
            if uploaded_file and uploaded_file.filename != '':
                # Save the uploaded image
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
                uploaded_file.save(file_path)

                # Preprocess the image for prediction
                img = load_img(file_path, target_size=(224, 224))  # Update with your model's input size
                img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

                # Perform prediction
                predictions = model.predict(img_array)
		
                predicted_class_index = np.argmax(predictions[0])  # Get class index with highest probability
		
                predicted_class = CLASS_NAMES[predicted_class_index]
		  # Map to class name

                # Fetch disease details
                disease_info = DISEASE_INFO.get(predicted_class, {
                    "description": "No information available.",
                    "cause": "N/A",
                    "symptoms": ["N/A"],
                    "prevention": ["N/A"],
                    "treatment": ["N/A"]
                })

                # Pass image, prediction, and disease info to the template
                return render_template(
                    'detection.html',
                    title='Detection',
                    active='detection',
                    image_url=url_for('static',
filename=f'uploads/{uploaded_file.filename}'),
                    prediction=predicted_class,
                    description=disease_info.get("description", "N/A"),
                    cause=disease_info.get("cause", "N/A"),
                    symptoms=disease_info.get("symptoms", []),
                    prevention=disease_info.get("prevention", []),
                    treatment=disease_info.get("treatment", [])
                )
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('error.html', title='Error', active='error')  # Redirect to error page

    # Render the page for GET requests
    return render_template('detection.html', title='Detection', active='detection')
@app.route('/about')
def about():
    return render_template('about.html', title='About Us', active='about')

@app.route('/contact')
def contact():
    return render_template('contact.html', title='Contact Us', active='contact')

if __name__ == "__main__":
    app.run(debug=True)
