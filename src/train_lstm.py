import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import pickle
import os
import io
from contextlib import redirect_stdout
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Loading dataset...")
df = pd.read_csv('sampled_dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"Unique texts: {df['text'].nunique()}")

# Data preprocessing
print("Preprocessing data...")
texts = df['text'].astype(str).values
labels = df['label'].values

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(np.unique(labels_encoded))
print(f"Number of classes: {num_classes}")
print(f"Class distribution:\n{df['label'].value_counts()}")

# Tokenize and pad sequences
max_words = 5000  # Reduced vocabulary for smaller model
max_length = 100  # Reduced sequence length

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels_encoded, 
    test_size=0.2, random_state=42, stratify=labels_encoded
)

# # Build LSTM model with constraints to keep it small
# print("Building LSTM model...")
# model = Sequential([
#     Embedding(input_dim=max_words, output_dim=64),
#     LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3),
#     Dense(32, activation='relu'),
#     Dropout(0.3),
#     Dense(num_classes, activation='softmax')
# ])

# model.build(input_shape=(None, max_length))

# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )


print("Building improved LSTM model...")

model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_length),
    Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
    Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Build the model with input shape
model.build(input_shape=(None, max_length))

optimizer = Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Now this works fine
model.summary()

# Print model summary and capture it
f = io.StringIO()
with redirect_stdout(f):
    model.summary()
model_summary = f.getvalue()
# Only print summary without special characters
try:
    print(model_summary)
except UnicodeEncodeError:
    # On Windows, sometimes there are encoding issues with special characters
    print("Model summary saved to results file.")

# Train the model
print("Training model...")
history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=20,
    validation_data=(X_test, y_test),
    verbose=1
)

# Extract training history
training_loss = history.history['loss']
training_accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

# Evaluate model
print("\nEvaluating model...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_test, y_pred_classes)
clf_report = classification_report(y_test, y_pred_classes, 
                                   target_names=label_encoder.classes_.astype(str))
conf_matrix = confusion_matrix(y_test, y_pred_classes)

print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(clf_report)

print("\nConfusion Matrix:")
print(conf_matrix)

# Save the model
# Create artifacts directory if it doesn't exist
os.makedirs('artifacts', exist_ok=True)

model_filename = 'artifacts/sentiment_lstm_model.h5'
tokenizer_filename = 'artifacts/tokenizer.pkl'
label_encoder_filename = 'artifacts/label_encoder.pkl'

print(f"\nSaving model to {model_filename}...")
model.save(model_filename)

print(f"Saving tokenizer to {tokenizer_filename}...")
with open(tokenizer_filename, 'wb') as f:
    pickle.dump(tokenizer, f)

print(f"Saving label encoder to {label_encoder_filename}...")
with open(label_encoder_filename, 'wb') as f:
    pickle.dump(label_encoder, f)

# Check file sizes
model_size = os.path.getsize(model_filename) / (1024 * 1024)
tokenizer_size = os.path.getsize(tokenizer_filename) / (1024 * 1024)

print(f"\nModel file size: {model_size:.2f} MB")
print(f"Tokenizer file size: {tokenizer_size:.2f} MB")
print(f"Total size: {model_size + tokenizer_size:.2f} MB")

# Save results to results directory
print("\nSaving results to results directory...")
os.makedirs('results', exist_ok=True)

# Create a comprehensive results file
results_filename = 'results/training_results.txt'
with open(results_filename, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("LSTM SENTIMENT ANALYSIS - TRAINING RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("-"*80 + "\n")
    f.write("DATASET INFORMATION\n")
    f.write("-"*80 + "\n")
    f.write(f"Dataset shape: {df.shape}\n")
    f.write(f"Unique texts: {df['text'].nunique()}\n")
    f.write(f"Number of classes: {num_classes}\n")
    f.write(f"Class distribution:\n{df['label'].value_counts().to_string()}\n\n")
    
    f.write("-"*80 + "\n")
    f.write("MODEL ARCHITECTURE\n")
    f.write("-"*80 + "\n")
    f.write(model_summary)
    f.write("\n")
    
    f.write("-"*80 + "\n")
    f.write("TRAINING HISTORY\n")
    f.write("-"*80 + "\n")
    f.write(f"Number of epochs: {len(training_loss)}\n")
    f.write(f"Batch size: 32\n\n")
    f.write("Final Training Metrics:\n")
    f.write(f"  Training Loss: {training_loss[-1]:.4f}\n")
    f.write(f"  Training Accuracy: {training_accuracy[-1]:.4f}\n")
    f.write(f"  Validation Loss: {val_loss[-1]:.4f}\n")
    f.write(f"  Validation Accuracy: {val_accuracy[-1]:.4f}\n\n")
    
    f.write("-"*80 + "\n")
    f.write("EVALUATION RESULTS\n")
    f.write("-"*80 + "\n")
    f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(clf_report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(conf_matrix))
    f.write("\n\n")
    
    f.write("-"*80 + "\n")
    f.write("MODEL FILES\n")
    f.write("-"*80 + "\n")
    f.write(f"Model file size: {model_size:.2f} MB\n")
    f.write(f"Tokenizer file size: {tokenizer_size:.2f} MB\n")
    f.write(f"Total size: {model_size + tokenizer_size:.2f} MB\n")
    f.write("\n")

print(f"Results saved to {results_filename}")

