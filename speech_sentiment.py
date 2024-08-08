from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import uvicorn
import librosa
import speech_recognition as sr
import requests
from io import BytesIO
import soundfile as sf
from starlette.requests import Request

app = FastAPI()
templates = Jinja2Templates(directory="templates")

HUGGINGFACE_API_KEY = "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"

class SentimentResult(BaseModel):
    label: str
    score: float

class AnalysisResult(BaseModel):
    sentiment: List[SentimentResult]
    summary: str

@app.post("/analyze-audio", response_model=AnalysisResult)
async def analyze_audio(file: UploadFile = File(...)):
    # Read audio file
    audio_bytes = await file.read()
    
    # Load audio file from bytes
    audio_data, rate = librosa.load(BytesIO(audio_bytes), sr=16000)
    
    # Convert audio to required format for speech recognition
    temp_audio_path = BytesIO()
    sf.write(temp_audio_path, audio_data, rate, format='WAV')
    temp_audio_path.seek(0)
    
    # Initialize recognizer and perform speech-to-text
    recognizer = sr.Recognizer()
    with sr.AudioFile(temp_audio_path) as source:
        audio = recognizer.record(source)
        transcription = recognizer.recognize_google(audio)

    # Perform sentiment analysis using Hugging Face API
    response = requests.post(
        "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base",
        headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
        json={"inputs": transcription}
    )
    sentiment_results = response.json()
    top_sentiment = max(sentiment_results[0], key=lambda x: x['score'])
    sentiment_label = top_sentiment['label']
    sentiment_score = top_sentiment['score']
    
    # Check if the length of the transcription is greater than 300 characters
    if len(transcription) > 300:
        # Perform summarization using Hugging Face API
        response = requests.post(
            "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
            headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
            json={"inputs": transcription}
        )
        summary_results = response.json()

        # Convert summary results to human readable format
        summary_text = summary_results[0]['summary_text']
    else:
        summary_text = ""

    return AnalysisResult(
        sentiment=[SentimentResult(label=sentiment_label, score=sentiment_score)],
        summary=summary_text
    )
# Mount the static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Single root endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)



