from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import librosa
import speech_recognition as sr
import requests
from io import BytesIO
import soundfile as sf
import subprocess
from tempfile import NamedTemporaryFile
from starlette.requests import Request

app = FastAPI()
templates = Jinja2Templates(directory="templates")

HUGGINGFACE_API_KEY = "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"

class SentimentResult(BaseModel):
    label: str
    score: float

class AnalysisResult(BaseModel):
    sentiment: Optional[List[SentimentResult]]
    summary: Optional[str]

@app.post("/analyze-audio", response_class=HTMLResponse)
async def analyze_audio(file: Optional[UploadFile] = File(None), video_link: Optional[str] = Form(None)):
    if file:
        # Read audio file
        audio_bytes = await file.read()
        
        # Load audio file from bytes
        audio_data, rate = librosa.load(BytesIO(audio_bytes), sr=16000)
    
    elif video_link:
        # Extract audio directly from the video URL using ffmpeg
        with NamedTemporaryFile(suffix=".wav") as temp_audio_file:
            # Use ffmpeg to stream the video and extract the audio
            ffmpeg_command = [
                "ffmpeg", "-i", video_link, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_audio_file.name
            ]
            subprocess.run(ffmpeg_command, check=True)
            
            # Load the extracted audio file
            audio_data, rate = librosa.load(temp_audio_file.name, sr=16000)
    
    else:
        return HTMLResponse(content="Please provide an audio file or video link.", status_code=400)
    
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
        "https://api-inference.huggingface.co/models/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
        json={"inputs": transcription}
    )
    sentiment_results = response.json()

    # Check if the API response is valid
    if isinstance(sentiment_results, list) and len(sentiment_results) > 0:
        top_sentiment = max(sentiment_results[0], key=lambda x: x['score'])
        sentiment_label = top_sentiment['label']
        sentiment_score = top_sentiment['score']
    else:
        sentiment_label = "Error"
        sentiment_score = 0.0

    # Check if the length of the transcription is greater than 300 characters
    if len(transcription) > 300:
        # Perform summarization using Hugging Face API
        response = requests.post(
            "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
            headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
            json={"inputs": transcription}
        )
        summary_results = response.json()

        # Convert summary results to human-readable format
        if isinstance(summary_results, list) and len(summary_results) > 0:
            summary_text = summary_results[0].get('summary_text', 'Error in generating summary.')
        else:
            summary_text = "Error in generating summary."
    else:
        summary_text = "Summary not needed."

    # Prepare the response content in text format
    response_content = (
        f"**Audio Analysis Report**\n\n"
        f"**Sentiment Analysis**\n"
        f"Label: {sentiment_label}\n"
        f"Score: {sentiment_score:.2f}\n\n"
        f"**Summary**\n"
        f"{summary_text}"
    )

    return HTMLResponse(content=response_content)

# Mount the static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Single root endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
