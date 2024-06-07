from __future__ import division, print_function
from flask import Flask, request, render_template,send_from_directory
import os
import cv2
from uuid import uuid4
from numpy import result_type
from signature import match
# coding=utf-8
import os
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
# from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Mach Threshold
THRESHOLD = 70

app = Flask(__name__)

MODEL_PATH = 'my_model.h5'
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    print(f"Input image path: {img_path}")  # Print the input path
    try:
        img = image.load_img(img_path, target_size=(150, 150))
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    prediction = model.predict(x)
    if prediction[0][0] > 0.7:
        return 'ORIGINAL'
    else:
        return 'FRAUDULENT'
    


imss= 'forgeries_1_5.png'


@app.route('/', methods=['GET', 'POST'])
def upload():
    
    if request.method == 'POST':
        file1 = request.files.get('file1')
        file2 = request.files.get('file2')
        
        if file1 and file2:
            # Generate unique filenames
            fname1 = secure_filename(file1.filename)
            fname2 = secure_filename(file2.filename)
    
            
            file1.save(os.path.join('uploads', fname1))
            file2.save(os.path.join('uploads', fname2))
            
            return checkSimilarity(fname1, fname2)
        
        return "File(s) missing"
    
    return render_template('index.html')



@app.route('/uploads/<filename>')
def serve_image(filename):
    return send_from_directory('uploads', filename)


def checkSimilarity(path1, path2):
    result = match(path1=path1, path2=path2)
    uploaded_file_path = os.path.join('uploads', path2)  # Construct the full path for the uploaded file
    prediction = model_predict(uploaded_file_path, model)  # Pass the uploaded file path to model_predict
    if prediction is None:
        msg = "Error loading input image"
    elif result <= THRESHOLD:
        msg = f"Failure: Signatures Do Not Match, Signatures are {result} % similar!!, and the new input is most likely {prediction}"
    else:
        msg = f"Success: Signatures Match, Signatures are {result} % similar!!, and the new input is most likely {prediction}"
    return render_template('compare.html', fname1=path1, fname2=path2, msg=msg)

if __name__ == '__main__':
    app.run()
    
