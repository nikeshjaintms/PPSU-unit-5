"""
app.py — ML/AI prediction API backend
--------------------------------------
Trains a tiny spam-detection model at startup and serves it through a REST API,
exactly matching the pattern described in "Unit 5: Displaying ML/AI Outputs on Web":

    fetch()  --->  Flask API  --->  JSON response { prediction, confidence }

Endpoints:
  GET /api/predict?text=...        -> single prediction
  GET /api/predictions             -> batch predictions for a demo inbox (used by the dashboard)
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import random

app = Flask(__name__)
CORS(app)  # allow the frontend page (opened as a local file / different port) to call this API

# ---------------------------------------------------------------------------
# 1. "Training" a tiny model (toy dataset, just to have a real ML pipeline)
# ---------------------------------------------------------------------------
training_texts = [
    "win money now", "free lottery prize claim now", "urgent click this link to win cash",
    "congratulations you won a free gift card", "act now limited offer free money",
    "hello how are you doing today", "let's meet for lunch tomorrow",
    "please review the attached report", "meeting rescheduled to 3pm",
    "can you send me the notes from class", "happy birthday hope you have a great day",
]
training_labels = [
    "Spam", "Spam", "Spam", "Spam", "Spam",
    "Not Spam", "Not Spam", "Not Spam", "Not Spam", "Not Spam", "Not Spam",
]

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_texts)
model = MultinomialNB()
model.fit(X_train, training_labels)


def predict_text(text: str):
    """Run the trained model on a single piece of text and return a JSON-ready dict."""
    X = vectorizer.transform([text])
    label = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = max(proba)
    return {"prediction": label, "confidence": round(float(confidence), 2)}


# ---------------------------------------------------------------------------
# 2. Routes
# ---------------------------------------------------------------------------

@app.route("/api/predict")
def predict():
    """Single prediction endpoint — mirrors section 5.1's example response shape."""
    text = request.args.get("text", "")
    if not text:
        return jsonify({"error": "Provide a 'text' query parameter"}), 400
    return jsonify(predict_text(text))


@app.route("/api/predictions")
def predictions():
    """
    Batch endpoint used by the dashboard (matches section 5.7's worked example).
    Simulates a small inbox of messages being scored by the model.
    """
    demo_inbox = [
        "win money now claim your prize",
        "let's meet for lunch tomorrow",
        "urgent click this link to win cash",
        "please review the attached report",
        "congratulations you won a free gift card",
        "meeting rescheduled to 3pm",
    ]
    random.shuffle(demo_inbox)  # so "Refresh" looks different each time
    results = []
    for i, text in enumerate(demo_inbox, start=1):
        pred = predict_text(text)
        results.append({
            "item": f"Email {i}",
            "label": pred["prediction"],
            "confidence": pred["confidence"],
        })
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
