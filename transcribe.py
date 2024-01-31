import whisperx
from utils import *
import gc 

def transcribe_audio_with_timestamps(audio_file = 'louvre_museum_tour.mp3'):
    '''
        This function only need be run once for transcribing the audio with 
        timestamps to obtain the respective json files
    '''
    device = "cpu" 
    batch_size = 4 # reduce if low on GPU mem
    compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("base", device, compute_type=compute_type)
    # save model to local path (optional)
    # model_dir = "/path/"
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)
    audio = whisperx.load_audio(audio_file)
    transcript = model.transcribe(audio, batch_size=batch_size)
    print(transcript) # before alignment
    write_list_to_file_as_json(transcript["segments"],'transcript.json')
    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=transcript["language"], device=device)
    transcript_time = whisperx.align(transcript["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    #print(transcript_time["segments"]) # after alignment
    write_list_to_file_as_json(transcript_time["segments"],'transcript_time.json')
    return

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
'''
diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs 
'''

if __name__=='__main__':
    transcribe_audio_with_timestamps()
    