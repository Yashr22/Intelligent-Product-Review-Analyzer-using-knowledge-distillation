import tensorflow as tf # Import TensorFlow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from starlette.middleware.cors import CORSMiddleware # Required for CORSMiddleware

# --- Configuration ---
MAX_SEQUENCE_LENGTH = 100 # IMPORTANT: Must match your training parameter
SENTIMENT_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# --- Global Asset Loading ---
# Initialize variables
MODEL = None
tokenizer = None

try:
    print("Loading LSTM model and tokenizer...")
    # Load the model using the imported function
    MODEL = load_model('sentiment_lstm_model.h5') 
    
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("Model and tokenizer loaded successfully.")
    
except Exception as e:
    print(f"FATAL ERROR: Could not load model or tokenizer. Prediction will fail. Details: {e}")
    # In a production environment, you would typically stop the server here.


app = FastAPI(title="LSTM Sentiment Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for frontend testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# New GET endpoint to confirm the server is running
@app.get("/")
def read_root():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "FastAPI server is running. Use /analyze-review for predictions."}


# Define the expected input structure for the API
class ReviewInput(BaseModel):
    review_text: str

# --- Prediction Endpoint ---
@app.post("/analyze-review")
def analyze(input_data: ReviewInput):
    # 1. Check if assets loaded successfully
    if MODEL is None or tokenizer is None:
        # Raise a 500 error if assets are missing
        raise HTTPException(status_code=500, detail="Server failed to load ML assets.")

    # 2. Pre-processing: Tokenize and Pad Sequence
    review = input_data.review_text
    
    # Tokenize the text using the loaded tokenizer
    sequence = tokenizer.texts_to_sequences([review])
    
    # Pad the sequence to the fixed length
    padded_sequence = pad_sequences(sequence, 
                                    maxlen=MAX_SEQUENCE_LENGTH, 
                                    padding='post')

    # 3. Prediction
    # The output will be an array like [[prob_neg, prob_neut, prob_pos]]
    # verbose=0 prevents progress bar output in the console
    raw_prediction = MODEL.predict(padded_sequence, verbose=0)
    
    # 4. Post-processing: Get the most likely class index
    # np.argmax finds the index (0, 1, or 2) with the highest probability
    predicted_index = np.argmax(raw_prediction, axis=1)[0]
    
    # Get the confidence (the probability of the predicted class)
    confidence = raw_prediction[0][predicted_index]
    
    # 5. Map index to label
    sentiment_label = SENTIMENT_MAP.get(predicted_index, "Unknown")
    
    # 6. Return the result
    return {
        "input_review": review,
        "sentiment": sentiment_label,
        "sentiment_code": int(predicted_index),
        "confidence": float(confidence)
    }
