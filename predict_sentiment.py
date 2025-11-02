import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from datetime import datetime
import io
from contextlib import redirect_stdout

class SentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.max_length = 100
        
    def load_model(self):
        """Load the trained model and tokenizer"""
        print("Loading model...")
        self.model = tf.keras.models.load_model('artifacts/sentiment_lstm_model.h5')
        
        with open('artifacts/tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)
            
        with open('artifacts/label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print("Model loaded successfully!")
        
    def predict(self, text):
        """
        Predict sentiment for a given text
        Args:
            text: Input text string
        Returns:
            dict: Predicted class and probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess text
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_length)
        
        # Predict
        prediction = self.model.predict(padded_sequence, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Get probabilities
        probabilities = {
            str(self.label_encoder.inverse_transform([i])[0]): float(prediction[0][i])
            for i in range(len(prediction[0]))
        }
        
        return {
            'predicted_class': int(predicted_class),
            'probabilities': probabilities,
            'confidence': float(prediction[0][predicted_class_idx])
        }

def main():
    # Example usage
    analyzer = SentimentAnalyzer()
    analyzer.load_model()
    
    # Test examples
    test_texts = [
        "This product is amazing! Highly recommend!",
        "Poor quality, very disappointed.",
        "It's okay, nothing special.",
        "Great quality and fast shipping.",
        "Terrible product. Waste of money.",
        "Average product, does the job."
    ]
    
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS PREDICTIONS")
    print("="*60)
    
    # Capture predictions for saving
    predictions_output = []
    
    for text in test_texts:
        result = analyzer.predict(text)
        
        # Map numeric labels to sentiment names
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment_name = sentiment_map[result['predicted_class']]
        
        output_line = f"\nText: '{text}'\n"
        output_line += f"Sentiment: {sentiment_name} (Class: {result['predicted_class']})\n"
        output_line += f"Confidence: {result['confidence']:.2%}\n"
        output_line += f"Probabilities: {result['probabilities']}\n"
        output_line += "-" * 60
        
        print(output_line)
        predictions_output.append({
            'text': text,
            'sentiment': sentiment_name,
            'class': result['predicted_class'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities']
        })
    
    # Save predictions to results directory
    print("\nSaving predictions to results directory...")
    os.makedirs('results', exist_ok=True)
    
    predictions_filename = 'results/predictions_results.txt'
    with open(predictions_filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SENTIMENT ANALYSIS - PREDICTION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Prediction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total Predictions: {len(predictions_output)}\n\n")
        
        for idx, pred in enumerate(predictions_output, 1):
            f.write("-"*80 + "\n")
            f.write(f"Prediction {idx}\n")
            f.write("-"*80 + "\n")
            f.write(f"Text: '{pred['text']}'\n")
            f.write(f"Sentiment: {pred['sentiment']} (Class: {pred['class']})\n")
            f.write(f"Confidence: {pred['confidence']:.2%}\n")
            f.write(f"Probabilities:\n")
            for label, prob in pred['probabilities'].items():
                f.write(f"  Class {label}: {prob:.4f}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"Predictions saved to {predictions_filename}")

if __name__ == '__main__':
    main()

