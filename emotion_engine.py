# --- Import Required Libraries ---
import os
import sqlite3
import random
import json
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import threading
from plotly.graph_objs import Scatter, Figure
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import time

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Database Setup ---
DB_PATH = "maia_emotion_db.db"

def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # For easier data handling
    return conn

# --- Flask App Setup ---
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # Enable CORS for all routes

# --- Background Thread for System-Initiated Conversations ---
def start_initiated_responses():
    """
    Periodically checks emotional states and generates proactive responses.
    """
    while True:
        time.sleep(30)  # Every 30 seconds
        if emotion_clusters:
            # Determine the most active emotional state
            most_active = max(emotion_clusters.items(), key=lambda c: c[1]["weight"])
            color, cluster = most_active
            emotion = cluster["emotion"]

            # Generate and log a proactive response
            response = generate_text_response({color: cluster})
            logging.info(f"Maia's proactive response: {response}")

# Start the thread
threading.Thread(target=start_initiated_responses, daemon=True).start()

def fetch_emotion_clusters():
    conn = connect_db()
    cursor = conn.cursor()
    logging.info("Fetching emotion clusters from the database.")
    cursor.execute("SELECT * FROM emotions")
    clusters = {row['color']: {
        "pleasure": row['pleasure'], 
        "arousal": row['arousal'], 
        "weight": row['weight'],
        "emotion": row['emotion'],
        "keywords": []
    } for row in cursor.fetchall()}
    
    # Populate keywords from the database
    cursor.execute("SELECT keyword, emotion FROM keywords")
    for row in cursor.fetchall():
        # Match the emotion in 'keywords' table to 'emotion' in 'clusters' dictionary
        color = next((c for c, e in clusters.items() if e["emotion"] == row['emotion']), None)
        if color:
            clusters[color]['keywords'].append(row['keyword'])
    
    conn.close()
    logging.info("Emotion clusters fetched successfully.")
    return clusters

emotion_clusters = fetch_emotion_clusters()

# Function to load the model
def load_model():
    global model
    try:
        model = tf.keras.models.load_model('emotion_recognition_model.h5')
        logging.info("Emotion recognition model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

# Load the model when the app starts
load_model()

def predict_emotion(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  # Create batch axis
    img_array = img_array / 255.0  # Normalize the image

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    
    # Map class indices to emotion labels (adjust this based on your training data)
    emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'trust', 'anticipation']
    predicted_emotion = emotion_labels[class_index]

    return predicted_emotion

def analyze_emotion(input_text, image_path=None):
    """Analyze input text and/or image for emotional content and activate corresponding clusters."""
    activations = {color: {"count": 0, "pleasure": 0, "arousal": 0} for color in emotion_clusters}
    
    # Text analysis
    if input_text:
        words = input_text.lower().split()
        conn = connect_db()
        cursor = conn.cursor()
        logging.info(f"Analyzing text: {input_text}")
        for word in words:
            cursor.execute("SELECT emotion, color, pleasure, arousal FROM connections WHERE keyword = ? OR connected_keyword = ?", (word, word))
            for row in cursor.fetchall():
                color = row['color']
                activations[color]['count'] += 1
                activations[color]['pleasure'] += row['pleasure'] or 0  # Handle NULL values
                activations[color]['arousal'] += row['arousal'] or 0  # Handle NULL values
        conn.close()
        logging.info("Text analysis complete.")

    # Image analysis
    if image_path:
        try:
            predicted_emotion = predict_emotion(image_path)
            logging.info(f"Image analysis result: {predicted_emotion}")
            
            # Map the predicted emotion to the corresponding cluster
            conn = connect_db()
            cursor = conn.cursor()
            cursor.execute("SELECT color, pleasure, arousal FROM emotions WHERE emotion = ?", (predicted_emotion,))
            row = cursor.fetchone()
            if row:
                color = row['color']
                activations[color]['count'] += 1
                activations[color]['pleasure'] += row['pleasure'] or 0
                activations[color]['arousal'] += row['arousal'] or 0
            conn.close()
        except Exception as e:
            logging.error(f"Error in image emotion prediction: {e}")
    
    # Update weights based on both text and image analysis
    for color in activations:
        if activations[color]['count'] > 0:
            activations[color]['weight'] = emotion_clusters[color]['weight'] * activations[color]['count']

    logging.info("Emotion analysis complete.")
    return activations

def update_cluster_weights(activations):
    """Adjust weights of clusters based on activations."""
    for color, activation in activations.items():
        if activation['count'] > 0:
            emotion_clusters[color]['weight'] += activation['weight']
    logging.info("Cluster weights updated.")

def generate_emotion_visualization():
    """Visualize current cluster activations as an interactive scatter plot."""
    colors = list(emotion_clusters.keys())
    pleasure = [cluster['pleasure'] for cluster in emotion_clusters.values()]
    arousal = [cluster['arousal'] for cluster in emotion_clusters.values()]
    weights = [cluster['weight'] for cluster in emotion_clusters.values()]
    
    fig = Figure(data=[
        Scatter(
            x=pleasure,
            y=arousal,
            mode='markers',
            marker=dict(
                size=[w * 20 for w in weights],
                color=colors,
                opacity=0.6
            ),
            text=colors
        )
    ])
    fig.update_layout(
        title="Emotional States: Pleasure vs Arousal",
        xaxis_title="Pleasure",
        yaxis_title="Arousal",
    )
    fig.write_html("static/emotion_visualization.html")
    logging.info("Emotion visualization generated.")

# --- Feedback Loop ---
def emotional_feedback_loop(input_text, image_path=None):
    activations = analyze_emotion(input_text, image_path)
    update_cluster_weights(activations)
    generate_emotion_visualization()
    return {"activations": activations}

# --- Frontend Integration ---
@app.route("/")
def serve_html():
    """Serve the main HTML file."""
    return send_from_directory('templates', 'index.html')

@app.route("/analyze", methods=["POST"])
def analyze():
    import tempfile
    input_text = request.form.get("text", "")
    image = request.files.get('image')
    temp_image_path = None

    if image:
        temp_image_path = os.path.join(tempfile.gettempdir(), image.filename)
        image.save(temp_image_path)  # Save the image temporarily for analysis

    results = emotional_feedback_loop(input_text, temp_image_path)

    if temp_image_path and os.path.exists(temp_image_path):
        os.remove(temp_image_path)  # Clean up the temporary file

    return jsonify(results)

@app.route("/visualize", methods=["GET"])
def visualize():
    """Generate and display a visualization of current emotional states."""
    generate_emotion_visualization()
    return jsonify({"message": "Visualization updated", "path": "/static/emotion_visualization.html"})

# --- Deployment ---
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)