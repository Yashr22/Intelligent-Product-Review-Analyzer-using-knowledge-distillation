import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import io
from contextlib import redirect_stdout
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("KNOWLEDGE DISTILLATION - LSTM SENTIMENT ANALYSIS")
print("="*80)

# Load data
print("\nLoading dataset...")
df = pd.read_csv('sampled_dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"Unique texts: {df['text'].nunique()}")

# Data preprocessing
print("\nPreprocessing data...")
texts = df['text'].astype(str).values
labels = df['label'].values

# Load label encoder (use the same one as teacher model)
with open('artifacts/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

labels_encoded = label_encoder.transform(labels)
num_classes = len(np.unique(labels_encoded))
print(f"Number of classes: {num_classes}")
print(f"Class distribution:\n{df['label'].value_counts()}")

# Load tokenizer (use the same one as teacher model)
print("\nLoading tokenizer...")
with open('artifacts/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_words = 5000
max_length = 100

# Tokenize sequences
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels_encoded, 
    test_size=0.2, random_state=42, stratify=labels_encoded
)

# Load teacher model
print("\nLoading teacher model...")
teacher_model = tf.keras.models.load_model('artifacts/sentiment_lstm_model.h5')
teacher_model.trainable = False  # Freeze teacher model

# Get soft targets from teacher model (with temperature)
print("Generating soft targets from teacher model...")
def softmax_with_temperature(logits, temperature):
    """Apply temperature scaling to logits before softmax"""
    logits = logits / temperature
    return tf.nn.softmax(logits)

temperature = 5.0  # Higher temperature produces softer probabilities

train_logits = teacher_model.predict(X_train, verbose=0)
train_soft_targets = softmax_with_temperature(train_logits, temperature).numpy()

test_logits = teacher_model.predict(X_test, verbose=0)
test_soft_targets = softmax_with_temperature(test_logits, temperature).numpy()

print(f"Soft targets generated with temperature={temperature}")
print(f"Train soft targets shape: {train_soft_targets.shape}")
print(f"Soft target distribution (first sample): {train_soft_targets[0]}")

# Build student model (smaller than teacher)
print("\nBuilding student model...")
student_model = Sequential([
    Embedding(input_dim=max_words, output_dim=32, name='student_embedding'),  # Half the dimension
    LSTM(16, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, name='student_lstm'),  # Half the units
    Dense(8, activation='relu', name='student_dense1'),  # Half the units
    Dropout(0.2, name='student_dropout'),
    Dense(num_classes, activation='linear', name='student_output')  # Linear for knowledge distillation
])

# Build the student model first to count parameters
student_model.build(input_shape=(None, max_length))

# Print student model summary
f = io.StringIO()
with redirect_stdout(f):
    student_model.summary()
student_model_summary = f.getvalue()
try:
    print(student_model_summary)
except UnicodeEncodeError:
    print("Student model summary saved to results file.")

# Count parameters
# Teacher model is already built/loaded
student_params = student_model.count_params()
# Try to count teacher params
try:
    teacher_params = teacher_model.count_params()
except:
    # If not built yet, estimate or use fixed value
    teacher_params = 170000  # Approximate based on architecture

compression_ratio = teacher_params / student_params if student_params > 0 else 0

print(f"\nModel Comparison:")
print(f"Teacher model parameters: {teacher_params:,}")
print(f"Student model parameters: {student_params:,}")
print(f"Compression ratio: {compression_ratio:.2f}x smaller")

# Knowledge distillation loss function
def distillation_loss(y_true, y_pred):
    """
    Combines hard labels loss and soft labels loss
    """
    # Hard labels loss (cross entropy with true labels)
    hard_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    
    # Soft labels loss (KL divergence with teacher predictions)
    # Note: soft_loss will be computed in custom training loop
    return hard_loss

# Custom training function for knowledge distillation
def train_with_knowledge_distillation(student_model, teacher_soft_targets, 
                                       true_labels, epochs=10, batch_size=64,
                                       alpha=0.7, temperature=5.0):
    """
    Train student model with knowledge distillation
    
    Args:
        student_model: The smaller student model to train
        teacher_soft_targets: Soft targets from teacher model
        true_labels: True labels for the data
        epochs: Number of training epochs
        batch_size: Batch size
        alpha: Weight for soft loss (1-alpha for hard loss)
        temperature: Temperature for softmax
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    losses = []
    accuracies = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0
        
        # Shuffle data
        indices = np.random.permutation(len(true_labels))
        X_shuffled = X_train[indices]
        y_hard_shuffled = true_labels[indices]
        y_soft_shuffled = teacher_soft_targets[indices]
        
        # Train in batches
        for i in range(0, len(true_labels), batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y_hard = y_hard_shuffled[i:i+batch_size]
            batch_y_soft = y_soft_shuffled[i:i+batch_size]
            
            with tf.GradientTape() as tape:
                # Forward pass
                logits = student_model(batch_X, training=True)
                
                # Hard loss (cross entropy with true labels)
                hard_loss = tf.keras.losses.sparse_categorical_crossentropy(
                    batch_y_hard, logits, from_logits=True
                )
                hard_loss = tf.reduce_mean(hard_loss)
                
                # Soft loss (KL divergence with teacher soft targets)
                student_probs = softmax_with_temperature(logits, temperature)
                # Manually compute KL divergence: sum(p * log(p/q))
                soft_loss = tf.reduce_sum(batch_y_soft * tf.math.log(batch_y_soft / (student_probs + 1e-8)), axis=1)
                soft_loss = tf.reduce_mean(soft_loss)
                
                # Combined loss
                total_loss = (1 - alpha) * hard_loss + alpha * (temperature ** 2) * soft_loss
                
                # Compute accuracy
                predictions = tf.nn.softmax(logits)
                pred_classes = tf.argmax(predictions, axis=1)
                accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(pred_classes, batch_y_hard), tf.float32)
                )
                
            # Backward pass
            gradients = tape.gradient(total_loss, student_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))
            
            epoch_loss += total_loss
            epoch_acc += accuracy
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        
        losses.append(avg_loss.numpy())
        accuracies.append(avg_acc.numpy())
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {avg_acc:.4f}")
    
    return losses, accuracies

# Train student model with knowledge distillation
print("\nTraining student model with knowledge distillation...")
print(f"Parameters: alpha=0.7, temperature={temperature}")
distillation_losses, distillation_accuracies = train_with_knowledge_distillation(
    student_model=student_model,
    teacher_soft_targets=train_soft_targets,
    true_labels=y_train,
    epochs=5,
    batch_size=64,
    alpha=0.7,
    temperature=temperature
)

# Evaluate student model
print("\nEvaluating student model...")
student_predictions = student_model.predict(X_test, verbose=0)
student_pred_classes = np.argmax(student_predictions, axis=1)

student_accuracy = accuracy_score(y_test, student_pred_classes)
print(f"\nStudent Model Test Accuracy: {student_accuracy:.4f}")

# Compare with teacher
teacher_predictions = teacher_model.predict(X_test, verbose=0)
teacher_pred_classes = np.argmax(teacher_predictions, axis=1)
teacher_accuracy = accuracy_score(y_test, teacher_pred_classes)
print(f"Teacher Model Test Accuracy: {teacher_accuracy:.4f}")
print(f"Accuracy difference: {(teacher_accuracy - student_accuracy):.4f}")

# Get classification reports
print("\nStudent Model Classification Report:")
clf_report_student = classification_report(y_test, student_pred_classes, 
                                          target_names=label_encoder.classes_.astype(str))
print(clf_report_student)

print("\nStudent Model Confusion Matrix:")
conf_matrix_student = confusion_matrix(y_test, student_pred_classes)
print(conf_matrix_student)

# Save student model with softmax activation for inference
print("\nSaving student model...")
os.makedirs('artifacts', exist_ok=True)
student_model_path = 'artifacts/student_lstm_model.h5'

# Build inference model with softmax
student_inference = Sequential([
    Embedding(input_dim=max_words, output_dim=32, name='student_embedding'),
    LSTM(16, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, name='student_lstm'),
    Dense(8, activation='relu', name='student_dense1'),
    Dropout(0.2, name='student_dropout'),
    Dense(num_classes, activation='softmax', name='student_output')
])

# Ensure inference model is built
student_inference.build(input_shape=(None, max_length))

# --- FIX: Use name-based weight copy for robust transfer ---
print("Copying trained student weights to inference model...")
for trained_layer in student_model.layers:
    if trained_layer.weights:  # Only transfer weights for layers that have them
        try:
            # Look up the layer in the inference model by its name
            inference_layer = student_inference.get_layer(name=trained_layer.name)
            # Transfer weights
            inference_layer.set_weights(trained_layer.get_weights())
        except Exception as e:
            # Log any potential issues but continue
            print(f"  WARNING: Could not copy weights for layer {trained_layer.name}. Error: {e}")
# --- END FIX ---

# # Copy weights from trained student model to inference model
# for i in range(len(student_model.layers) - 1):
#     student_inference.layers[i].set_weights(student_model.layers[i].get_weights())

# # Copy the last layer's weights
# last_layer_weights = student_model.layers[-1].get_weights()
# student_inference.layers[-1].set_weights(last_layer_weights)

# Save the inference model
student_inference.save(student_model_path)
print(f"Student model saved to {student_model_path}")

# Check file sizes
teacher_size = os.path.getsize('artifacts/sentiment_lstm_model.h5') / (1024 * 1024)
student_size = os.path.getsize(student_model_path) / (1024 * 1024)

print(f"\nModel file sizes:")
print(f"Teacher model: {teacher_size:.2f} MB")
print(f"Student model: {student_size:.2f} MB")
print(f"Size reduction: {(teacher_size - student_size):.2f} MB ({((1 - student_size/teacher_size) * 100):.1f}% smaller)")

# Save results to results directory
print("\nSaving knowledge distillation results...")
os.makedirs('results', exist_ok=True)

results_filename = 'results/knowledge_distillation_results.txt'
with open(results_filename, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("KNOWLEDGE DISTILLATION - RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("-"*80 + "\n")
    f.write("DATASET INFORMATION\n")
    f.write("-"*80 + "\n")
    f.write(f"Dataset shape: {df.shape}\n")
    f.write(f"Unique texts: {df['text'].nunique()}\n")
    f.write(f"Number of classes: {num_classes}\n\n")
    
    f.write("-"*80 + "\n")
    f.write("MODEL COMPARISON\n")
    f.write("-"*80 + "\n")
    f.write(f"Teacher Model Parameters: {teacher_params:,}\n")
    f.write(f"Student Model Parameters: {student_params:,}\n")
    f.write(f"Compression Ratio: {compression_ratio:.2f}x smaller\n\n")
    
    f.write("-"*80 + "\n")
    f.write("STUDENT MODEL ARCHITECTURE\n")
    f.write("-"*80 + "\n")
    f.write(student_model_summary)
    f.write("\n")
    
    f.write("-"*80 + "\n")
    f.write("KNOWLEDGE DISTILLATION SETTINGS\n")
    f.write("-"*80 + "\n")
    f.write(f"Temperature: {temperature}\n")
    f.write(f"Alpha (soft loss weight): 0.7\n")
    f.write(f"Epochs: 15\n")
    f.write(f"Batch size: 32\n\n")
    
    f.write("-"*80 + "\n")
    f.write("TRAINING RESULTS\n")
    f.write("-"*80 + "\n")
    f.write(f"Final Training Loss: {distillation_losses[-1]:.4f}\n")
    f.write(f"Final Training Accuracy: {distillation_accuracies[-1]:.4f}\n\n")
    
    f.write("-"*80 + "\n")
    f.write("EVALUATION RESULTS\n")
    f.write("-"*80 + "\n")
    f.write(f"Teacher Model Test Accuracy: {teacher_accuracy:.4f}\n")
    f.write(f"Student Model Test Accuracy: {student_accuracy:.4f}\n")
    f.write(f"Accuracy Difference: {teacher_accuracy - student_accuracy:.4f}\n\n")
    
    f.write("Student Model Classification Report:\n")
    f.write(clf_report_student)
    f.write("\n\nStudent Model Confusion Matrix:\n")
    f.write(str(conf_matrix_student))
    f.write("\n\n")
    
    f.write("-"*80 + "\n")
    f.write("MODEL FILES\n")
    f.write("-"*80 + "\n")
    f.write(f"Teacher model size: {teacher_size:.2f} MB\n")
    f.write(f"Student model size: {student_size:.2f} MB\n")
    f.write(f"Size reduction: {(teacher_size - student_size):.2f} MB ({((1 - student_size/teacher_size) * 100):.1f}% smaller)\n")
    f.write("\n")

print(f"Results saved to {results_filename}")
print("\n" + "="*80)
print("KNOWLEDGE DISTILLATION COMPLETED!")
print("="*80)

