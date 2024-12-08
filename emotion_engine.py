# --- Import Required Libraries ---
import os
import sqlite3
import random
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import datetime

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flask App Setup ---
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# --- Database Setup ---
DB_PATH = "maia_emotion_db.db"

def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --- Create Memories Table ---
def initialize_database():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT NOT NULL,
            content TEXT NOT NULL,
            emotion TEXT NOT NULL,
            intensity REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()
    logging.info("Database initialized with 'memories' table.")

initialize_database()

# --- Load Emotion Clusters ---
def fetch_emotion_clusters():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM emotions")
    clusters = {row['color']: {
        "pleasure": row['pleasure'], 
        "arousal": row['arousal'], 
        "weight": row['weight'],
        "emotion": row['emotion'],
        "keywords": []
    } for row in cursor.fetchall()}

    # Populate keywords
    cursor.execute("SELECT keyword, emotion FROM keywords")
    for row in cursor.fetchall():
        color = next((c for c, e in clusters.items() if e["emotion"] == row['emotion']), None)
        if color:
            clusters[color]['keywords'].append(row['keyword'])

    conn.close()
    return clusters

emotion_clusters = fetch_emotion_clusters()

# --- Memory Storage ---
def form_memory(event_type, content, emotion, intensity):
    """
    Store interactions persistently in the memories database.
    """
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO memories (event, content, emotion, intensity) 
        VALUES (?, ?, ?, ?)
    """, (event_type, content, emotion, intensity))
    conn.commit()
    conn.close()
    logging.info(f"Memory stored: {event_type}, {content}, {emotion}, {intensity}")

# --- Emotional Analysis ---
def analyze_emotion(input_text):
    conn = connect_db()
    cursor = conn.cursor()
    activations = {color: {"count": 0, "pleasure": 0, "arousal": 0} for color in emotion_clusters}

    if input_text:
        words = input_text.lower().split()
        for word in words:
            cursor.execute("SELECT emotion, color, pleasure, arousal FROM connections WHERE keyword = ?", (word,))
            row = cursor.fetchone()
            if row:
                color = row['color']
                activations[color]['count'] += 1
                activations[color]['pleasure'] += row['pleasure'] or 0
                activations[color]['arousal'] += row['arousal'] or 0
            else:
                cursor.execute("INSERT INTO connections (keyword, emotion) VALUES (?, ?)", (word, "neutral"))
                logging.info(f"New keyword '{word}' added to database.")

    conn.commit()
    conn.close()

    for color in activations:
        if activations[color]['count'] > 0:
            activations[color]['weight'] = emotion_clusters[color]['weight'] * activations[color]['count']

    return activations

# --- Memory-Based Response Generation ---
def retrieve_relevant_memories(emotion, limit=5):
    """
    Retrieve the most relevant past memories based on emotion.
    """
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT content, intensity, timestamp 
        FROM memories 
        WHERE emotion = ? 
        ORDER BY intensity DESC, timestamp DESC 
        LIMIT ?
    """, (emotion, limit))
    memories = cursor.fetchall()
    conn.close()
    return memories

def generate_text_response(activations, input_text=""):
    """
    Generate a contextual response referencing past memories.
    """
    if not activations:
        return "I'm calm and neutral. What's on your mind?"

    most_active = max(activations.items(), key=lambda item: item[1]["count"])
    color, data = most_active
    emotion = emotion_clusters[color]["emotion"]

    # Fetch relevant memories
    relevant_memories = retrieve_relevant_memories(emotion)

    # Memory-based response templates
    response_templates = {
        "joy": "I recall a time when {memory}—it felt truly wonderful!",
        "sadness": "Thinking back to {memory}, I remember how things felt heavy but manageable.",
        "anger": "There was a time when {memory} really tested my patience.",
        "trust": "I recall {memory}, a moment when trust meant everything.",
    }

    if relevant_memories:
        selected_memory = random.choice(relevant_memories)["content"]
        memory_response = response_templates.get(emotion, "I remember {memory}.")
        response = memory_response.format(memory=selected_memory)
    else:
        response = f"I’m feeling {emotion}. Let's explore this further."

    # Contextual expansion
    if input_text:
        response += f" Your words resonated deeply: '{input_text}'."

    return response

# --- Analyze Endpoint ---
@app.route("/analyze", methods=["POST"])
def analyze():
    input_text = request.form.get("text", "")
    activations = analyze_emotion(input_text)
    most_active_emotion = max(activations.items(), key=lambda x: x[1]["count"], default=(None, None))[0]

    # Persist memory if an emotion is recognized
    if most_active_emotion:
        form_memory("user_interaction", input_text, emotion_clusters[most_active_emotion]["emotion"], activations[most_active_emotion]["count"])

    # Generate a contextual response
    text_response = generate_text_response(activations, input_text)

    return jsonify({
        "activations": activations,
        "text_response": text_response,
        "confirmation": f"Your input affected these emotional states: {', '.join(activations.keys())}."
    })

# --- Deployment ---
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
