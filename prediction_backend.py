from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os


import numpy as np


from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.spatial.distance import cosine


app = Flask(__name__)

# Load the trained model
MODEL_PATH = "pneumonia_cnn_model_v2.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found. Please check the path and try again.")

model = tf.keras.models.load_model(MODEL_PATH)

# Directory to save uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed image formats
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif"}

# Image size used during training
IMG_SIZE = 150

def allowed_file(filename):
    """Check if the file has a valid image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_pneumonia(image_path):
    """Predict if the given X-ray image has pneumonia or not."""
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for model input
    
    prediction = model.predict(img_array)[0][0]  # Get the prediction score
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100

    return {
        "prediction": "Pneumonia" if prediction > 0.5 else "Normal",
        "confidence": round(float(confidence), 2),  # Convert NumPy float32 to Python float
    }





def extract_features(image_path):
    model = VGG16(weights="imagenet", include_top=False, pooling="avg")
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    features = model.predict(img_array)
    return features.flatten()

def compare_deep_features(image_path, reference_xray_path, threshold=0.4):
    features1 = extract_features(image_path)
    features2 = extract_features(reference_xray_path)

    similarity = 1 - cosine(features1, features2)
    print(f"Cosine Similarity: {similarity:.2f}")
    return similarity > threshold



@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]



    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file format. Allowed: PNG, JPG, JPEG, BMP, GIF"}), 400

    elif not compare_deep_features(file_path, "NORMAL2-IM-1427-0001.jpeg", 0.5):
        print("\n\n\n",file_path,"\n\n\n")
        if compare_deep_features(file_path, "person1946_bacteria_4874.jpeg", 0.5):
            return jsonify(predict_pneumonia(file_path))    
        else:
            return jsonify({"error": "Unrelated image"}), 401
    


    
    # Run prediction
    result = predict_pneumonia(file_path)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
