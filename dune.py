# code for playing audio, pausing asking user question, transcribing and LLM logic
import pyaudio, wave, whisperx
from openai import OpenAI
from pynput import keyboard
from RealtimeTTS import OpenAIEngine, TextToAudioStream
import os 

os.environ['OPENAI_API_KEY']="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"
client = OpenAI(
    # This is the default and can be omitted
    api_key="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",
)
#elevenlabs.set_api_key("ed0b37bd0196e46583e6d87094b2794e")

system_prompt = {
    'role': 'system', 
    'content': 'be concise and informative, dont ever use emojis or special chars since im passing output to tts'}

device = "cpu"
batch_size = 4  # reduce if low on GPU mem
compute_type = "int8"
model, answer, history = whisperx.load_model("base", device, compute_type=compute_type), "", []
engine = OpenAIEngine() # replace with your TTS engine
stream_openai = TextToAudioStream(engine)

# Define stop_recording at the global level
stop_recording = False


def generate_llm_response(messages):
    global answer
    answer = ""
    for chunk in client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, stream=True):
        if (text_chunk := chunk.choices[0].delta.content):
            answer += text_chunk
            print(text_chunk, end="", flush=True) 
            yield text_chunk

def on_press(key):
    global stop_recording # put stream.isplaying condition
    if key == keyboard.Key.space:
        stop_recording = not stop_recording

listener = keyboard.Listener(on_press=on_press)
listener.start()

while True:
    print("\n\nTap space when you're ready to speak. ", end="", flush=True)
    
    # Wait for the user to press space to start recording
    while not stop_recording:
        pass  

    print("Recording... Tap space when you're done.")
    audio, frames = pyaudio.PyAudio(), []
    stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=512)
    
    # Record until the space is pressed again to stop
    while stop_recording: 
        frames.append(stream.read(512))

    stream.stop_stream(), stream.close(), audio.terminate()

    # Transcribe recording using whisper
    with wave.open("voice_record.wav", 'wb') as wf:
        wf.setparams((1, pyaudio.PyAudio().get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
        wf.writeframes(b''.join(frames))
    user_text = " ".join(sentence['text'] for sentence in model.transcribe("voice_record.wav", batch_size)['segments'])
    print(f'>>>{user_text}\n<<< ', end="", flush=True)
    history.append({'role': 'user', 'content': user_text})

    # Generate and stream output
    generator = generate_llm_response([system_prompt] + history[-10:])
    stream_openai.feed(generator)
    stream_openai.play()
    history.append({'role': 'assistant', 'content': answer})

