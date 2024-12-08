# --- Import Required Libraries ---
import os
import sqlite3
import random
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

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

# --- Initialize Database ---
def initialize_database():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS connections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT NOT NULL,
            emotion TEXT NOT NULL,
            color TEXT NOT NULL,
            pleasure REAL DEFAULT 0.5,
            arousal REAL DEFAULT 0.5,
            weight REAL DEFAULT 1.0
        );
    """)
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
    logging.info("Database initialized with 'connections' and 'memories' tables.")

initialize_database()

# --- Fetch Emotion Clusters ---
def fetch_emotion_clusters():
    conn = connect_db()
    cursor = conn.cursor()

    # Initialize clusters
    cursor.execute("SELECT * FROM connections")
    clusters = {
        row['color']: {
            "emotion": row['emotion'],
            "pleasure": row['pleasure'],
            "arousal": row['arousal'],
            "weight": row['weight'],
            "keywords": []
        } for row in cursor.fetchall()
    }

    # Populate keywords
    cursor.execute("SELECT keyword, emotion FROM connections")
    for row in cursor.fetchall():
        color = next((c for c, e in clusters.items() if e["emotion"] == row['emotion']), None)
        if color:
            clusters[color]['keywords'].append(row['keyword'])

    conn.close()
    return clusters

emotion_clusters = fetch_emotion_clusters()

# --- Memory Storage ---
def store_memory(event_type, content, emotion, intensity):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO memories (event, content, emotion, intensity) 
        VALUES (?, ?, ?, ?)
    """, (event_type, content, emotion, intensity))
    conn.commit()
    conn.close()
    logging.info(f"Memory stored: {event_type}, {content}, {emotion}, {intensity}")

def analyze_emotion(input_text):
    conn = connect_db()
    cursor = conn.cursor()

    # Initialize emotion activations
    activations = {color: {"count": 0, "pleasure": 0, "arousal": 0} for color in emotion_clusters if color}
    unknown_keywords = []  # Collect unknown keywords

    if input_text:
        words = input_text.lower().split()
        for word in words:
            cursor.execute("SELECT emotion, color, pleasure, arousal FROM connections WHERE keyword = ?", (word,))
            row = cursor.fetchone()

            if row:
                color = row['color'] or "gray"  # Ensure color defaults to a valid value
                # Use default value 0 if any field is None
                pleasure = row['pleasure'] or 0
                arousal = row['arousal'] or 0

                # Update activation counts
                activations[color]['count'] += 1
                activations[color]['pleasure'] += pleasure
                activations[color]['arousal'] += arousal
            else:
                # Handle unknown keywords with default emotion
                unknown_keywords.append(word)
                default_emotion = {
                    "pleasure": 0.5,
                    "arousal": 0.5,
                    "color": "#808080",
                    "emotion": "neutral"
                }

                # Insert into the database with default values
                cursor.execute("""
                    INSERT INTO connections (keyword, emotion, color, pleasure, arousal) 
                    VALUES (?, ?, ?, ?, ?)
                """, (word, default_emotion["emotion"], default_emotion["color"], 
                      default_emotion["pleasure"], default_emotion["arousal"]))
                conn.commit()
                logging.info(f"New keyword '{word}' added with default emotions.")

    conn.close()

    # Calculate activation weights
    for color in activations:
        if activations[color]['count'] > 0:
            activations[color]['weight'] = emotion_clusters.get(color, {}).get('weight', 1) * activations[color]['count']

    return activations, unknown_keywords

# --- Memory-Based Response Generation ---
def retrieve_memories(emotion, limit=5):
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

    # Handle None values safely in max() operations
    most_active = max(
        ((color, data) for color, data in activations.items() if color),
        key=lambda item: item[1]["count"],
        default=(None, None)
    )
    color, data = most_active

    if not color or not data:
        return "I'm neutral and here to listen. Share more if you like."

    emotion = emotion_clusters.get(color, {}).get("emotion", "neutral")

    # Fetch relevant memories
    relevant_memories = retrieve_memories(emotion)

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


# --- Flask Routes ---
@app.route("/", methods=["GET"])
def home():
    return send_from_directory("templates", "index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    input_text = request.form.get("text", "").strip()
    logging.info(f"Received input: {input_text}")

    if not input_text:
        return jsonify({"error": "Input text cannot be empty."}), 400

    try:
        activations, unknown_keywords = analyze_emotion(input_text)
        logging.info(f"Emotion activations: {activations}")
        logging.info(f"Unknown keywords: {unknown_keywords}")

        if unknown_keywords:
            unknown_word = unknown_keywords[0]
            question = f"What does '{unknown_word}' mean? Could you explain it for me?"
            return jsonify({
                "activations": activations,
                "question": question,
                "unknown_keywords": unknown_keywords
            })

        most_active_emotion = max(activations.items(), key=lambda x: x[1]["count"], default=(None, None))[0]
        if most_active_emotion:
            store_memory("user_interaction", input_text, emotion_clusters[most_active_emotion]["emotion"], activations[most_active_emotion]["count"])

        text_response = generate_text_response(activations, input_text)
        logging.info(f"Generated response: {text_response}")

        return jsonify({
            "activations": activations,
            "text_response": text_response,
            "confirmation": f"Your input affected these emotional states: {', '.join(activations.keys())}."
        })

    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        return jsonify({"error": "An error occurred during analysis."}), 500

# --- Deployment ---
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)