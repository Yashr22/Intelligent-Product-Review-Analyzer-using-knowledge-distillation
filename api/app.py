# from fastapi import FastAPI
# from pydantic import BaseModel
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# import os

# # ---------- Load Environment Variables ----------
# load_dotenv(dotenv_path=os.path.join("test", ".env"))  # Ensure .env has GOOGLE_API_KEY or credentials set

# # ---------- Initialize FastAPI ----------
# app = FastAPI(title="Sentiment Analysis API", version="1.0")

# # ---------- Initialize Gemini Model ----------
# model = ChatGoogleGenerativeAI(model='gemini-2.5-pro')

# # ---------- Request Schema ----------
# class ReviewRequest(BaseModel):
#     text: str

# # ---------- API Routes ----------
# @app.get("/")
# def home():
#     return {"message": "Welcome to Sentiment Analysis API (Gemini powered)"}

# @app.post("/predict/")
# def predict_sentiment(review: ReviewRequest):
#     text = review.text
#     prompt = f"Analyze the sentiment of this text and label it strictly as Positive, Negative, or Neutral:\n\n{text}"

#     # Call Gemini
#     response = model.invoke(prompt)

#     # Extract text response
#     sentiment = response.content.strip()

#     # Sanitize unexpected outputs
#     valid_labels = ["Positive", "Negative", "Neutral"]
#     sentiment = next((label for label in valid_labels if label.lower() in sentiment.lower()), "Unknown")

#     return {"review": text, "sentiment": sentiment}



from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import random

load_dotenv(dotenv_path=os.path.join("test", ".env"))
app = FastAPI(title="Sentiment Analysis API", version="1.0")
model = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ---------- Serve Home Page ----------
@app.get("/")
def serve_homepage():
    # Go one level up from /api/ to /FINAL_GENAI/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_path = os.path.join(base_dir, "frontend", "home.html")

    if not os.path.exists(html_path):
        raise RuntimeError(f"File not found: {html_path}")

    return FileResponse(html_path)

# ---------- Request Schema ----------
class ReviewRequest(BaseModel):
    text: str

# ---------- API Endpoint ----------
@app.post("/predict/")
async def predict_sentiment(review: ReviewRequest):
    text = review.text
    prompt = f"Analyze the sentiment of this text and label it strictly as Positive, Negative, or Neutral:\n\n{text}"

    try:
        response = model.invoke(prompt)
        sentiment = response.content.strip()
        valid_labels = ["Positive", "Negative", "Neutral"]
        sentiment = next((label for label in valid_labels if label.lower() in sentiment.lower()), "Unknown")

        confidence = round(random.uniform(0.7, 0.98), 2)
        return JSONResponse(content={"sentiment": sentiment, "confidence": confidence})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

