from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import requests
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
    
    # Send the raw audio data to the Hugging Face API
    response = requests.post(
        "https://api-inference.huggingface.co/models/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
        data=audio_bytes
    )
    sentiment_results = response.json()

    # Check if the API response is valid
    if isinstance(sentiment_results, list) and len(sentiment_results) > 0:
        top_sentiment = max(sentiment_results, key=lambda x: x['score'])
        sentiment_label = top_sentiment['label']
        sentiment_score = top_sentiment['score']
    else:
        sentiment_label = "Error"
        sentiment_score = 0.0

    # Prepare the response content in text format
    response_content = (
        f"**Audio Analysis Report**\n\n"
        f"**Sentiment Analysis**\n"
        f"Label: {sentiment_label}\n"
        f"Score: {sentiment_score:.2f}\n\n"
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
