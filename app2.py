import os
import cv2
import base64
import numpy as np
import pickle
import requests
import subprocess
import time
import torch
from gfpgan import GFPGANer
from flask import Flask, request, jsonify
from flask_cors import CORS
from insightface.app import FaceAnalysis  # Import ArcFace
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

# Initialize the Flask application
app = Flask(__name__)
CORS(app,origins=["http://localhost:8081"])

GFPGAN_MODEL_PATH = "C:/Users/sanka/Desktop/AI ML project/server/scripts"


# Initialize ArcFace and MTCNN
faceApp = FaceAnalysis(providers=['CPUExecutionProvider'])  # Use CPU for processing
faceApp.prepare(ctx_id=0, det_size=(160, 160))
detector = MTCNN()

gfpgan = GFPGANer(
    model_path=GFPGAN_MODEL_PATH,
    upscale=4,  # Upscale factor (2x, 4x, etc.)
    arch="clean",  # Model architecture
    channel_multiplier=2,  # Channel multiplier
    bg_upsampler=None,  # Background upsampler (optional)
    device="cpu",
)

requests.get("https://ams-server-1.onrender.com")

# Define the path to the SVM model file
svm_model_file = 'svm_model.pkl'

# Check if the SVM model file exists
if not os.path.exists(svm_model_file):
    raise FileNotFoundError(f"SVM model file '{svm_model_file}' not found")

with open(svm_model_file, 'rb') as f:
    svm_model = pickle.load(f)


# In-memory storage for image chunks and processed images
image_chunks = defaultdict(list)
processed_images = {}

def get_embedding(face_img):
    # Get embedding using ArcFace
    face_img = face_img.astype('float32')
    face_img = cv2.resize(face_img, (112, 112))  # ArcFace standard input size
    embedding = faceApp.get(face_img)[0].embedding
    return embedding

# Function to process the image chunks and detect faces
def process_image_chunks(email):
    try:
        image_base64 = ''.join(image_chunks.pop(email))
        image_base64 += '=' * (-len(image_base64) % 4)  # Padding to make length a multiple of 4
        image_data = base64.b64decode(image_base64)
        image = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        print("Image decoded successfully", flush=True)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Enhance the image using GFPGAN
        _, _, restored_image = gfpgan.enhance(
            image_rgb,
            has_aligned=False,  # Set to False for entire image enhancement
            only_center_face=False,  # Set to False for entire image enhancement
            paste_back=True,  # Paste the enhanced result back to the original image
        )

        # Check if the output is valid
        if restored_image is None:
            print("Error: GFPGAN returned an empty result.", flush=True)
            return None

        # Convert enhanced image back to BGR for saving
        restored_image_bgr = cv2.cvtColor(restored_image, cv2.COLOR_RGB2BGR)

        # Save the enhanced image
        enhanced_image_path = os.path.join("test images\\", f"{email}_enhanced_image.jpg")
        cv2.imwrite(enhanced_image_path, restored_image_bgr)
        print(f"Enhanced image saved: {enhanced_image_path}", flush=True)
        
        # Detect face
        results = detector.detect_faces(restored_image_bgr)
        if not results:
            print("No faces detected", flush=True)
            return None
        
        # Process each detected face
        predictions = []
        confidence_threshold = 0.90  # Confidence threshold for individual predictions

        for result in results:
            print(f"Detected face with confidence {result['confidence']}", flush=True)
            if result['confidence'] < confidence_threshold:
                continue
    
            x1, y1, w, h = result['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + w, y1 + h
            face = image[y1:y2, x1:x2]
            face = cv2.resize(face, (160, 160))
    
    # Get embedding
            embedding = get_embedding(face)
    
    # Get decision function for all classes
            decision_function = svm_model.decision_function([embedding])
            print("decision funtion is ", decision_function)
            max_confidence_score = np.max(decision_function)
            second_max_confidence_score = np.partition(decision_function[0], -2)[-2]
    
    # Check if the difference between the top two scores is significant
            if (max_confidence_score - second_max_confidence_score) >= 0.005 * max_confidence_score:
                if max_confidence_score >= confidence_threshold:
                    # Assuming embedding is the input embedding for prediction
                    # Perform the prediction using the trained SVM model
                    prediction = svm_model.predict([embedding])
                    # Print the prediction (which is the class label predicted by the model)
                    print("Prediction:", prediction)
                    # The prediction is already the label, so you don't need to use encoder.inverse_transform.
                    # If the labels in your embeddings.pkl file are directly the classes you want, you can use the prediction as the label.
                    predicted_label = prediction[0]
                    # Append the predicted label to your predictions list
                    predictions.append(predicted_label)

                else:
                    predictions.append("Unknown")
            else:
                predictions.append("Unknown")
        
        print(f"Predictions: {predictions}", flush=True)
        return predictions

    except Exception as e:
        print(f"Error in process_image_chunks: {e}", flush=True)
        return None

# Endpoint for face recognition
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        email = data.get('email')
        chunk = data.get('chunk')
        sequence_number = data.get('sequenceNumber')
        is_last_chunk = data.get('isLastChunk')
        is_last_image = data.get('isLastImage')

        print(f"Received data for email: {email}, sequence_number: {sequence_number}, is_last_chunk: {is_last_chunk}, is_last_image: {is_last_image}", flush=True)

        # Store image chunks
        if email not in image_chunks:
            image_chunks[email] = []

        image_chunks[email].append(chunk)

        # If last chunk received, process the image
        if is_last_chunk:
            predictions = process_image_chunks(email)
            if predictions:
                if email not in processed_images:
                    processed_images[email] = []
                processed_images[email].extend(predictions)

        # If last image received, return processed predictions
        if is_last_image:
            predictions = processed_images.pop(email, [])
            return jsonify({"predictions": predictions, "key": 2}), 200

        return jsonify({"key": 1}), 200
    except Exception as e:
        print(f"Error in predict endpoint: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

# Function to stop ngrok if already running
def stop_ngrok():
    try:
        subprocess.run(['ngrok', 'terminate', 'all'])
        print("ngrok terminated successfully", flush=True)
    except Exception as e:
        print(f"Error terminating ngrok: {e}", flush=True)

# Stop any existing ngrok process
stop_ngrok()

# Start ngrok manually using the configuration file in the background
ngrok_process = subprocess.Popen(['ngrok', 'start', '--config', 'ngrok.yml', 'my_tunnel'])

# Allow some time for ngrok to start
time.sleep(15)

# Function to get the public URL from ngrok
def get_ngrok_url(retries=10, wait=2):
    for attempt in range(retries):
        try:
            response = requests.get('http://localhost:4040/api/tunnels')
            data = response.json()
            public_url = data['tunnels'][0]['public_url']
            print(f"ngrok public URL fetched on attempt {attempt + 1}: {public_url}", flush=True)
            return public_url
        except (requests.ConnectionError, IndexError) as e:
            print(f"Attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(wait)
    raise ConnectionError("Failed to connect to ngrok API")

def send_ngrok_url(ngrok_url):
    # Send the ngrok URL to the Render server
    render_server_url = "https://ams-server-1.onrender.com/store-url"
    response = requests.post(render_server_url, json={"ngrok_url": ngrok_url, "naunce": "$Ams999"})
    return response.json()

# Retrieve the ngrok public URL
try:
    ngrok_url = get_ngrok_url()
    print(f"ngrok URL: {ngrok_url}", flush=True)
    
    response = send_ngrok_url(ngrok_url)
    
    while(response['key'] != 1):
        response = send_ngrok_url(ngrok_url)
    
    if response['key'] == 1:
        print("ngrok URL sent to Render server successfully", flush=True)
    else:
        print("Failed to send ngrok URL to Render server", flush=True)

finally:
    print("Initialization complete", flush=True)

if __name__ == '__main__':
    app.run(debug=True)
