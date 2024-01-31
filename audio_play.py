import pygame, os, threading
from pynput import keyboard
import pyaudio
import wave
import whisperx
from openai import OpenAI
from RealtimeTTS import OpenAIEngine, TextToAudioStream
from utils import *
# API KEYS -------
os.environ['OPENAI_API_KEY']="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"
# -----------------

class AudioTour:
    def __init__(self, audio_file_path, llm_client, tts_engine):
        self.audio_file_path = audio_file_path
        self.playing = False
        self.paused = False
        self.recording_event = threading.Event()
        self.llm_client = llm_client
        self.tts_engine = tts_engine
        self.stream_openai = TextToAudioStream(tts_engine)
        self.device = "cpu"
        self.batch_size = 4  # reduce if low on GPU mem
        self.compute_type = "int8"
        self.model, self.history = whisperx.load_model("base", "cpu", compute_type="int8"), []
        
        self.initialize_llm_prompt()
        
        self.start_keyboard_listener()
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file_path)
        
    def start_keyboard_listener(self):
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
    
    def initialize_llm_prompt(self):
        self.history.append({'role': 'system', 'content': '''you are a museum tour guide you will be given some context about 
                             whatever you have told the user using which you have to answer the user's question in the most
                             concise way as possible max 30 words.'''})
        return

    def on_press(self, key):
        if key == keyboard.Key.space:
            print(f"Space bar pressed. Playing: {self.playing}, Paused: {self.paused}")
            if self.playing and not self.paused:
                pygame.mixer.music.pause()
                self.paused = True
                timestamp = pygame.mixer.music.get_pos() / 1000.0
                self.transcript = self.fetch_transcript_up_to_timestamp(timestamp)
                print(f"Paused at {timestamp}s. Transcript: {self.transcript}")
                self.start_recording()
            elif not self.recording_event.is_set():
                self.recording_event.set()  # Signal to stop recording
            return
        
    def start_playback(self):
        pygame.mixer.music.play()
        self.playing = True
        self.paused = False
        
    def start_recording(self):
        self.recording_event.clear()  # Reset the event
        threading.Thread(target=self.record_and_transcribe_query).start()
        
        
    def stop_recording(self):
            self.recording = False
            
    def fetch_transcript_up_to_timestamp(self, timestamp):
        # Implement the transcript fetching logic here (using utils.py functionality)
        transcript = read_transcript_and_get_text('transcript_time.json', timestamp)
        return transcript

    def handle_user_query(self, user_query):
        print("Please speak your query after the beep.")
        transcript_prompt = {'role':'assistant','content':self.transcript}
        self.history.append(transcript_prompt)
        #user_query = self.record_and_transcribe_query()
        self.history.append({'role': 'user', 'content': user_query})
        #full_context = {"transcript": transcript, "user_query": user_query}
        ai_response = self.generate_llm_response(self.history)
        self.play_ai_response(ai_response)
        return
    
    def record_and_transcribe_query(self):
        # Implement the recording and transcription logic here
        print("Recording... Tap space when you're done.")
        audio, frames = pyaudio.PyAudio(), []
        stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=512)
        
        # Record until the space is pressed again to stop
        while not self.recording_event.is_set():
            frames.append(stream.read(512))


        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        audio.terminate()

        # Save the recording to a file
        with wave.open("voice_record.wav", 'wb') as wf:
            wf.setparams((1, pyaudio.PyAudio().get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
            wf.writeframes(b''.join(frames))

        # Transcribe the recording
        user_text = " ".join(sentence['text'] for sentence in self.model.transcribe("voice_record.wav", self.batch_size)['segments'])
        print(f'>>>{user_text}\n<<< ', end="", flush=True)
        self.handle_user_query(user_text)
        return user_text

    def generate_llm_response(self,messages):
        self.current_llm_answer = ""
        for chunk in llm_client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, stream=True):
            if (text_chunk := chunk.choices[0].delta.content):
                self.current_llm_answer += text_chunk
                print(text_chunk, end="", flush=True) 
                yield text_chunk

    def play_ai_response(self, llm_generator):
        # Use TextToAudioStream to play the AI response
        self.stream_openai.feed(llm_generator)
        self.stream_openai.play()
        # once entire audio ai response is played, add ai response to history and clear variables
        self.history.append({'role':'assistant','content':self.current_llm_answer})
        #self.current_llm_answer =""
        self.start_playback()
        return

# Initialization
llm_client = OpenAI(
    # This is the default and can be omitted
    api_key="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"
)
tts_engine = OpenAIEngine()  # Replace with your TTS engine initialization
audio_tour = AudioTour('louvre_museum_tour.mp3',llm_client, tts_engine)
audio_tour.start_playback()

# Keep the script running
while True:
    pass
