# code for playing audio, pausing asking user question, transcribing and LLM logic
import pyaudio, wave, whisperx
import pygame
from openai import OpenAI
from pynput import keyboard
from RealtimeTTS import OpenAIEngine, TextToAudioStream
from utils import *
import os, time 

def generate_llm_response(messages):
    global answer
    answer = ""
    for chunk in client.chat.completions.create(model="gpt-4-1106-preview", messages=messages, stream=True):
        if (text_chunk := chunk.choices[0].delta.content):
            answer += text_chunk
            print(text_chunk, end="", flush=True) 
            yield text_chunk

def on_press(key):
    global stop_recording # put stream.isplaying condition
    if key == keyboard.Key.space:
        stop_recording = not stop_recording

def play_audio(audio_file, timestamp):
    # Initialize Pygame mixer and play audio from the given timestamp
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    print("timestamp: ",timestamp)
    pygame.mixer.music.play(start=timestamp)

    # Non-blocking loop that checks the stop_recording flag
    while not stop_recording:
        pass

    # Pause the audio and get the current playback position
    pygame.mixer.music.pause()
    new_timestamp = pygame.mixer.music.get_pos() / 1000.0  # Convert milliseconds to seconds
    print("pause timestamp: ",new_timestamp + timestamp)
    return int(new_timestamp + timestamp)

def conversation(n,system_prompt,transcript_prompt,history):
    for i in range(n):
        print("\n\nTap space when you're ready to speak. ", end="", flush=True)
        
        # Wait for the user to press space to start recording
        #while not stop_recording:
        #    pass  

        print("Recording... Tap space when you're done.")
        audio, frames = pyaudio.PyAudio(), []
        stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=512)
        
        # Record until the space is pressed again to stop
        while stop_recording: 
            frames.append(stream.read(512))

        stream.stop_stream(), stream.close(), audio.terminate()

        # Transcribe recording using whisper
        s =time.time()
        with wave.open("voice_record.wav", 'wb') as wf:
            wf.setparams((1, pyaudio.PyAudio().get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
            wf.writeframes(b''.join(frames))
        user_text = " ".join(sentence['text'] for sentence in model.transcribe("voice_record.wav", batch_size)['segments'])
        print(f'>>>{user_text}\n<<< ', end="", flush=True)
        history.append({'role': 'user', 'content': user_text})
        t = time.time()
        print("time to transcribe this was : ",t-s)
        # Generate and stream output
        generator = generate_llm_response([system_prompt] + [transcript_prompt] + history[-10:])
        stream_openai.feed(generator)
        stream_openai.play()
        history.append({'role': 'assistant', 'content': answer})
        
    return history

if __name__=="__main__":
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    os.environ['OPENAI_API_KEY']="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"
    client = OpenAI(
        # This is the default and can be omitted
        api_key="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",
    )
    #elevenlabs.set_api_key("ed0b37bd0196e46583e6d87094b2794e")
    device = "cpu"
    batch_size = 4  # reduce if low on GPU mem
    compute_type = "int8"
    model, answer = whisperx.load_model("tiny", device, compute_type=compute_type), ""
    engine = OpenAIEngine() # replace with your TTS engine
    stream_openai = TextToAudioStream(engine)
    # Define stop_recording at the global level
    stop_recording = False

    system_prompt = {
        'role': 'system', 
        'content': '''you are a museum tour guide and you are talking live with a person, find the context given below,
                     that is all the information you have given the user about museum, use it to answer user's question in 
                     a short, friendly and human way. dont use emojis or any text in your responses that a human can't speak.
                     be concise and to the point in your answer. after your answer,
                     end it by saying that if there are no more question, lets continue with our tour...'''
    }
    
    history = []
    timestamp = 1000
    while True:
        timestamp = play_audio("louvre_museum_tour.mp3",timestamp)
        transcript = read_transcript_and_get_text('transcript_time.json', timestamp)
        transcript_prompt = {
            'role':'assistant',
            'content':transcript
        }
        history = conversation(1,system_prompt,transcript_prompt,history)
        time.sleep(500/1000)
        
        