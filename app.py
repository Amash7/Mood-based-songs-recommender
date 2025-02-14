import gradio as gr
import whisper
import torch
import google.generativeai as genai

#Loading Whisper Model's Base version
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base").to(device)

#Setting up API for songs generation from transcription
GOOGLE_GEMINI_API_KEY = "AIzaSyDpVlnMJf9mRl3VQO83lXuNzershODM7cs"
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

def analyze_mood_and_suggest_songs(text):

    prompt = f"""
    You are an AI assistant that recommends songs based on mood. 
    A user said: '{text}'. 
    Identify their mood and suggest 10 songs in the same language.
    Return only the song list without explanations.
    """

    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        
        if response and hasattr(response, "text"):
            return response.text.strip()
        else:
            return "⚠️ No valid song recommendations were generated."
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

def process_audio(audio_path):

    if not audio_path:
        return "⚠️ Please upload an audio file."

    try:
        #Transcribing
        transcription = model.transcribe(audio_path)["text"]
        
        #Songs from the transcribed text
        song_recommendations = analyze_mood_and_suggest_songs(transcription)

        return f"🎤 Transcription: {transcription}\n\n🎶 Recommended Songs:\n{song_recommendations}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

#Gradio app GUI
app = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Speak About Your Mood"),
    outputs=gr.Textbox(label="🎶 Recommended Songs", interactive=False, lines=6),
    title="🌙✨ **سرگوشی - Mood-Based Song Recommender** ✨🌙",
    description="🎙️ Speak about your **mood**, and this AI will **recommend 10 songs** in your language!<br>💡 **Powered by Whisper & Google Gemini** 💡",
    theme="soft",  
    css="""
        body {
            background: linear-gradient(to right, #141E30, #243B55);
            color: white;
        }
        .gradio-container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 20px;
        }
        .gr-button {
            background: linear-gradient(90deg, #ff6a00, #ee0979);
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 12px;
        }
        .gr-button:hover {
            background: linear-gradient(90deg, #ee0979, #ff6a00);
            transform: scale(1.05);
        }
        .gr-textbox {
            background: rgba(255, 255, 255, 0.2);
            color: white;
            border-radius: 8px;
            padding: 10px;
        }
        h1 {
            color: #fff;
            text-align: center;
            font-size: 28px;
        }
    """
)
#Driver part
app.launch(share=True)
