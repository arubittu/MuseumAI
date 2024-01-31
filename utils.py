import json

def write_list_to_file_as_json(lst, filename):
    """
    Writes the contents of a list to a specified text file in JSON format.
    """
    with open(filename, 'w') as file:
        json.dump(lst, file)

def read_list_from_file_as_json(filename):
    """
    Reads the contents of a file assumed to be in JSON format and returns it as a list.
    """
    with open(filename, 'r') as file:
        return json.load(file)

def read_transcript_and_get_text(file_path, time_in_seconds):
    """
    Reads the transcript JSON file, and returns the transcript text before the given time in seconds.

    :param file_path: Path to the transcript JSON file.
    :param time_in_seconds: Time in seconds before which the transcript is required.
    :return: Transcript text or error message.
    """
    try:
        with open(file_path, 'r') as file:
            transcript_data = json.load(file)
    except FileNotFoundError:
        return "Error: The file was not found."
    except json.JSONDecodeError:
        return "Error: The file is not a valid JSON."

    if not isinstance(time_in_seconds, (int, float)) or time_in_seconds < 0:
        return "Error: Invalid time input. Time should be a non-negative number."
    elif time_in_seconds > transcript_data[-1]['end']:
        return "Error: Invalid time input. Time should be lesser than total Audio time"

    return get_transcript_before_time(transcript_data, time_in_seconds)

def get_transcript_before_time(transcript_data, time):
    """
    Returns the transcript text of all words spoken before the given time in seconds.
    Efficiently stops processing once the input time is exceeded.
    """
    transcript = ""
    for entry in transcript_data:
        for word_info in entry['words']:
            # Check if word has timing information
            if 'start' in word_info and 'end' in word_info:
                if word_info['end'] > time:
                    return transcript.strip()
                transcript += word_info['word'] + " "
            else:
                # For words without timing, include them until a timed word exceeds the input time
                transcript += word_info['word'] + " "
    return transcript.strip()

# Example usage:
# transcript = read_transcript_and_get_text('/path/to/transcript_time.json', 10.0)
# print(transcript)

# Example usage:
# my_list = [{'word': 'Au', 'start': 3350.436, 'end': 3350.476, 'score': 0.0}, ...]
# write_list_to_file_as_json(my_list, 'data.json')
# loaded_list = read_list_from_file_as_json('data.json')
