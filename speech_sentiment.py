import speech_recognition as sr
import requests
from fastapi import FastAPI

app = FastAPI()

#sentiment_API = "https://api-inference.huggingface.co/models/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
sentiment_API = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
headers = {"Authorization": "Bearer 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"}
recognizer = sr.Recognizer()

def query(payload):
	response = requests.post(sentiment_API, headers=headers, json=payload)
	return response.json()

def record_audio():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source, phrase_time_limit=5)
    return audio

def recognize_speech(audio):
    try:
        text = recognizer.recognize_google(audio)
        sentiment = query({"inputs": text})
        sentiment = sentiment[0][0]['label']
        print(f"Sentiment: {sentiment}")

    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
    except sr.RequestError:
        print("Sorry, there was an error processing your request.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    audio = record_audio()
    recognize_speech(audio)