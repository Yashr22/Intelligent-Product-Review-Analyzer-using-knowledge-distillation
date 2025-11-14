from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import os


# ---------- Initialize FastAPI ----------
app = FastAPI(title="Sentiment Analysis API", version="1.0")

# ---------- Load Model and Tokenizer ----------
MODEL_PATH = "../artifacts/student_lstm_model.h5"
TOKENIZER_PATH = "../artifacts/tokenizer.pkl"


# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# ---------- Load Tokenizer ----------
with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)


max_len = 100  # same as used during training

# ---------- Request Schema ----------
class ReviewRequest(BaseModel):
    text: str

# ---------- API Routes ----------
@app.get("/")
def home():
    return {"message": "Welcome to Sentiment Analysis API 🚀"}

@app.post("/predict/")
def predict_sentiment(review: ReviewRequest):
    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([review.text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')

    # Predict sentiment
    prediction = model.predict(padded)
    sentiment = np.argmax(prediction, axis=1)[0]

    # Map numeric label to text
    labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    result = labels.get(sentiment, "Unknown")

    return {"review": review.text, "sentiment": result}
