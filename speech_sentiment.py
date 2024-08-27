from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
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
    sentiment: Optional[List[SentimentResult]]
    summary: Optional[str]

@app.post("/analyze-audio", response_class=HTMLResponse)
async def analyze_audio(file: UploadFile = File(...)):
    # Read audio file
    audio_bytes = await file.read()

    # Load the audio file from bytes
    with BytesIO(audio_bytes) as audio_buffer:
        audio_buffer.seek(0)
        audio_data, samplerate = sf.read(audio_buffer)

    # Convert audio data back to bytes for API processing
    with BytesIO() as temp_audio_path:
        sf.write(temp_audio_path, audio_data, samplerate, format='WAV')
        temp_audio_path.seek(0)

        # Step 1: Perform speech-to-text using Whisper
        transcription_response = requests.post(
            "https://api-inference.huggingface.co/models/openai/whisper-large-v3",
            headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
            data=temp_audio_path.read()
        )
        transcription_data = transcription_response.json()

        # Extract the transcription text
        transcription = transcription_data.get("text", "")

        # Step 2: Perform sentiment analysis using Hugging Face API directly on audio
        temp_audio_path.seek(0)
        sentiment_response = requests.post(
            "https://api-inference.huggingface.co/models/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
            data=temp_audio_path.read()
        )
        sentiment_results = sentiment_response.json()

        # Ensure that the sentiment analysis results are valid
        if isinstance(sentiment_results, list) and len(sentiment_results) > 0:
            top_sentiment = max(sentiment_results, key=lambda x: x['score'])
            sentiment_label = top_sentiment['label']
            sentiment_score = top_sentiment['score']
        else:
            sentiment_label = "Error"
            sentiment_score = 0.0

    # Step 3: Summarize the transcription if it's long enough
    summary_text = ""
    if len(transcription.split()) > 100:  # Check for word count > 100
        summary_response = requests.post(
            "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
            headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
            json={"inputs": transcription}
        )
        summary_results = summary_response.json()
        if isinstance(summary_results, list) and len(summary_results) > 0:
            summary_text = summary_results[0].get('summary_text', 'Error in generating summary.')
        else:
            summary_text = "Error in generating summary."

    # Prepare the response content in text format
    response_content = (
        f"Audio Analysis Report\n\n"
        f"Transcription\n{transcription}\n\n"
        f"Sentiment Analysis\n"
        f"Label: {sentiment_label}\n"
        f"Score: {sentiment_score:.2f}\n\n"
        f"Summary:\n{summary_text}"
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
