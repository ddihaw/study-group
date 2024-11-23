import tensorflow as tf

# Load model
model = tf.keras.models.load_model('Model.h5')

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

# Create instance for web app
app = Flask(__name__)

# Define image labels
class_labels = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

# Predict's endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get image from client request
    file = request.files['image']
    img = Image.open(file)

    # Preprocess Image (adjust to model input shape)
    img = img.resize((224, 224))  # Resizing to 224x224 (refer to model input shape)
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Predicting the image
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction) # Output: Index of image labels

    # Retrieve labels array using output of predicted_class
    predicted_class_label = class_labels[predicted_class]

    # Return result as JSON format
    return jsonify({'prediction': predicted_class_label})

# Start flask server
if __name__ == '__main__':
    app.run(debug=True)