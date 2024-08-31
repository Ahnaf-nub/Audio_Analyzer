# Audio Sentiment Analyzer with Transcription and Summarization
This FastAPI-based web application allows users to upload audio files for sentiment analysis, speech-to-text transcription, and automatic summarization. The app leverages state-of-the-art models from Hugging Face to analyze emotions directly from audio data and generate text transcriptions. If the transcribed text exceeds 100 words, the app will automatically summarize the content.
**Check it out at: https://audio-analyzer.vercel.app**
![image](https://github.com/user-attachments/assets/4eddb7cf-a74d-4e21-9068-3b75b6c06302)

### 🚀 Features
**Audio Upload: Upload audio files for analysis directly through the web interface.**
**Speech-to-Text Transcription: Converts spoken words to text using the Whisper-large-v3 model from Hugging Face.**
**Audio-Based Sentiment Analysis: Analyzes emotions in audio files using the wav2vec2-lg-xlsr-en-speech-emotion-recognition model.**
**Text Summarization: Automatically summarizes lengthy transcriptions using the facebook/bart-large-cnn model.**
**Real-time Feedback: View analysis results instantly on the web interface.**
### 🌟 How It Works
**Upload an Audio File: Use the web interface to upload your audio file.**
**Analyze: The application will transcribe the audio, analyze the sentiment, and summarize the text if it's over 100 words.**
**View Results: The analysis results, including the transcription, sentiment score, and summary, are displayed in the browser.**
### For running locally:
Clone the repository:
```
git clone https://github.com/Ahnaf-nub/Audio_Analyzer.git
cd Audio_Analyzer
```
Install the required Python packages:
```
pip install -r requirements.txt
```
Set up your Hugging Face API key:
```
export HUGGINGFACE_API_KEY='your_huggingface_api_key'
```
Run the application:
```
uvicorn app:app --reload
```
**Open your browser and navigate to http://127.0.0.1:8000 to start using the app.**
