from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Charger les modèles
classification_model = load_model('models/classification.h5')
segmentation_model = load_model('models/segmentation.h5')

CLASS_NAMES = ['bottle', 'can', 'chain', 'drink-carton', 'hook', 'propeller', 'shampoo-bottle', 'standing-bottle', 'tire', 'valve']
SEGMENTATION_SIZE = (128, 128)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_segmentation(image_path):
    original_img = cv2.imread(image_path)
    h, w = original_img.shape[:2]
    
    img = cv2.resize(original_img, SEGMENTATION_SIZE)
    img = img / 255.0
    mask = segmentation_model.predict(np.expand_dims(img, axis=0))[0]
    
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    red_mask = np.zeros((h, w, 3), dtype=np.uint8)
    red_mask[:, :, 2] = mask
    
    overlay = cv2.addWeighted(original_img, 0.7, red_mask, 0.3, 0)
    return overlay

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            
            # Classification
            img = cv2.imread(save_path)
            img = cv2.resize(img, (224, 224))
            pred = classification_model.predict(np.expand_dims(img, axis=0))
            class_name = CLASS_NAMES[np.argmax(pred)]
            
            # Segmentation
            result = process_segmentation(save_path)
            result_filename = 'result_' + filename
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            
            return render_template('index.html',
                                 class_name=class_name,
                                 original_img=filename,
                                 seg_img=result_filename)
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)