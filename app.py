from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import io

app = Flask(__name__)

def load_pretrained_model():
    return load_model('model.h5')

@app.route('/')
def home():
    return render_template('predict_disease2.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        loaded_model = load_pretrained_model()

        img = image.load_img(io.BytesIO(uploaded_file.read()), target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = loaded_model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        
        l1 = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']
        predicted_class_entity = l1[predicted_class_index]

        # Load disease information from JSON
        with open('disease_mapping.json', 'r') as f:
            disease_info = json.load(f)

        # Retrieve actions for the predicted disease
        actions = disease_info['diseases'].get(predicted_class_entity, [])

        return render_template('prediction_result.html', predicted_disease=predicted_class_entity, actions=actions)

if __name__ == '__main__':
    app.run(debug=True)
