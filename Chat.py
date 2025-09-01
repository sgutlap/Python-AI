from email.mime import message
import wave, struct, os
from pvrecorder import PvRecorder
from playsound import playsound
from IPython.display import Image, display
import google.generativeai as genai
from gtts import gTTS
import warnings
import speech_recognition as sr
import threading


warnings.filterwarnings("ignore", category=DeprecationWarning)

genai.configure(api_key="AIzaSyDKnu87M0x5IO7YiIug0p6wl0jWXizBPVc")
model = genai.GenerativeModel("gemma-3-27b-it")
#response = model.generate_content([
#   {"role": "model", "parts": "You are a witty assistant, always answering with a joke."},
#    {"role": "user", "parts": "Who are you?"}
#])
#
#print(response.text)

class Chatbot:
    def __init__(self, model):
        self.model = model
        self.history = [
            {"role": "model", "parts": "You are a helpful assistant."}
        ]

    def chat(self, message):
        self.history.append({"role": "user", "parts": message})

        response = self.model.generate_content(self.history)
        reply = response.text.strip()

        self.history.append({"role": "model", "parts": reply})

        print(f"Model: {reply}")
        self.speak(reply)

    def speak(self, message, index=0):
        def play_audio(path):
            playsound(path)
            os.remove(path)

        speech_file_path = os.path.join(os.getcwd(), f"speech_{index}.mp3")
        tts = gTTS(text=message, lang="en")
        tts.save(speech_file_path)

        threading.Thread(target=play_audio, args=(speech_file_path,), daemon=True).start()
        
    def record_audio(self, index=0):
        recorder = PvRecorder(device_index=-1, frame_length=512)
        audio = []
        filepath = os.path.join(os.getcwd(), f"recorded_{index}.wav")

        try:
            recorder.start()
            print("Recording... Press Ctrl+C to stop.")
            while True:
                frame = recorder.read()
                audio.extend(frame)
        except KeyboardInterrupt:
            print("Recording stopped.")
            recorder.stop()
            with wave.open(filepath, 'w') as f:
                f.setparams((1, 2, 16000, 0, "NONE", "NONE"))
                f.writeframes(struct.pack("<" + "h" * len(audio), *audio))
        finally:
            recorder.delete()
        return filepath

    def transcribe(self, audio_path):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "(Could not understand audio)"
        except sr.RequestError:
            return "(Speech recognition service error)"

    def voicechat(self):
        recorded_filepath = self.record_audio(index=len(self.history))
        message = self.transcribe(recorded_filepath)
        print(f"Transcribed: {message}")
        self.chat(message)


if __name__ == "__main__":
    chatbot = Chatbot(model)
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            break
        chatbot.chat(user_input)
