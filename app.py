# app.py
from flask import Flask, render_template, request, url_for
import numpy as np
import cv2
from keras.models import load_model
import os

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png'}

# Charger le modèle
MODEL = load_model('dog_breed.h5')

# Classes de chiens
CLASS_NAMES = ['scottish_deerhound', 'maltese_dog', 'bernese_mountain_dog', 'entlebucher']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_breed(image_path):
    # Lire et prétraiter l'image
    opencv_image = cv2.imread(image_path)
    opencv_image = cv2.resize(opencv_image, (224, 224))
    opencv_image = np.expand_dims(opencv_image, axis=0)
    
    # Faire la prédiction
    prediction = MODEL.predict(opencv_image)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_file = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        if file and allowed_file(file.filename):
            # Sauvegarder le fichier
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Faire la prédiction
            predicted_breed = predict_breed(filepath)
            
            return render_template('index.html', 
                                 prediction=predicted_breed,
                                 image_file=filename)
            
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)