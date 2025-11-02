# Sentiment Analysis LSTM Model

A lightweight LSTM model for sentiment analysis, optimized for a size of 5-10 MB.

## Project Structure

```
final_genai/
├── artifacts/              # Model artifacts (gitignored)
│   ├── sentiment_lstm_model.h5
│   ├── tokenizer.pkl
│   └── label_encoder.pkl
├── results/               # Training and prediction results
│   ├── training_results.txt
│   └── predictions_results.txt
├── data.csv               # Training dataset
├── train_lstm.py          # Training script
├── predict_sentiment.py   # Inference script
├── requirements.txt       # Dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Dataset
- **Source**: `data.csv`
- **Features**: Text reviews and sentiment labels (0: Negative, 1: Neutral, 2: Positive)

## Model Architecture
- **Embedding Layer**: 5000 vocabulary size, 64 dimensions
- **LSTM Layer**: 32 units with dropout (0.3)
- **Dense Layer**: 16 units with dropout (0.3)
- **Output Layer**: 3 classes (Negative, Neutral, Positive)

## Installation

```bash
pip install -r requirements.txt
```

## Training the Model

```bash
python train_lstm.py
```

This will:
1. Load and preprocess the dataset
2. Train the LSTM model
3. Evaluate performance
4. Save the model, tokenizer, and label encoder to the `artifacts/` folder
5. Display model size
6. Save detailed results to `results/training_results.txt`

## Making Predictions

```bash
python predict_sentiment.py
```

This will:
1. Load the trained model from `artifacts/`
2. Make predictions on example texts
3. Display results in console
4. Save detailed predictions to `results/predictions_results.txt`

Or use the `SentimentAnalyzer` class in your own code:

```python
from predict_sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()
analyzer.load_model()

result = analyzer.predict("This product is amazing!")
print(result)
```

## Model Performance
The model will display accuracy, classification report, and confusion matrix after training.

## Model Size
The model is optimized to be between 5-10 MB to ensure fast inference and easy deployment.

## Labels
- **0**: Negative sentiment
- **1**: Neutral sentiment
- **2**: Positive sentiment

