**PlantAI: Automated Plant Disease Detection using AI Vision Algorithm**

Intern Name: N Shiva Prasad Reddy

Internship Domain: Artificial Intelligence

Project Mentor: Revathi Venugari

Project Coordinator: Arathy Pillai

**Project Overview**

PlantAI is an AI-powered system designed to detect plant diseases based on leaf images. Using advanced computer vision techniques, this system helps farmers and agricultural professionals diagnose plant diseases quickly and accurately, providing real-time predictions for better crop management.

The project leverages deep learning algorithms and image processing techniques to analyze plant leaf images and identify the specific disease affecting the plant, making it a valuable tool for modern farming practices.

**Technology Stack**

**Frontend**: HTML, CSS

**Backend**: Python, Flask

**Deep Learning Framework**: TensorFlow, Keras

**Machine Learning Libraries**: NumPy, Pandas, Matplotlib, Seaborn

**Computer Vision Libraries**: OpenCV, Pillow

**Deployment**: Vercel

**Platform**: Google Colab


**Project Milestones**

**Milestone 1: Project Setup and Image Data Acquisition**

**Development Environment Setup**: Set up the development environment and version control using Git for smooth collaboration and code management.

**Image Upload Mechanism:** Developed a Flask-based interface to allow users to upload plant leaf images, with format validation for common image types (PNG, JPEG).

**Image Preprocessing**: Enhanced the quality of uploaded images by applying OpenCV and Pillow for noise reduction, resizing, and normalization. The preprocessing includes:

**Noise Reduction:** Using filters like Gaussian Blur.

**Resizing:** Images are resized to a standard dimension suitable for model input (e.g., 224x224).

**Normalization**: Pixel values scaled to the range [0, 1] for faster and more efficient model training.


**Milestone 2: Image Segmentation and Feature Extraction**

**Image Segmentation:** Isolated the plant region in each image using OpenCV techniques like thresholding and edge detection.

**Feature Extraction:** Extracted important features such as color, texture, and shape using libraries like NumPy and Pandas, which are crucial for training the disease detection model.

**Model Training:** Developed an initial Convolutional Neural Network (CNN) using TensorFlow and Keras, and trained it on labeled datasets to identify plant diseases.

**Milestone 3: Disease Classification and User Interface**

**Disease Classification:** Refined and finalized the CNN model for accurate classification of plant diseases based on input images.

**Model Validation:** Evaluated model performance using metrics like precision, recall, and F1-score, and optimized it using hyperparameter tuning to improve accuracy.

**User Interface:** Built a web-based interface using Flask for seamless interaction with the model, enabling users to upload images and receive disease predictions.
