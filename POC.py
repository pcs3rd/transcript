import io
import os
import speech_recognition as sr
import whisper
import torch
import subprocess
import nltk
from nltk.tokenize import sent_tokenize
import csv

from argparse import ArgumentParser
from datetime import datetime, timedelta, UTC
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
from faster_whisper import WhisperModel


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--device", default="auto", help="device to user for CTranslate2 inference. Not implementerd",
                        choices=["auto", "cuda","cpu"])                   
    parser.add_argument("--compute_type", default="auto", help="Type of quantization to use",
                        choices=["auto", "int8", "int8_floatt16", "float16", "int16", "float32"])
    parser.add_argument("--translation_lang", default='English',
                        help="Which language should we translate into." , type=str)
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--threads", default=0,
                        help="number of threads used for CPU inference", type=int)
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
                        
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float) 
    if 'linux' in platform:
            parser.add_argument("--default_microphone", default='pulse',
                                help="Default microphone name for SpeechRecognition. "
                                    "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    def censor(text, forbidden):
        return ' '.join(
        word if word.casefold() not in forbidden else '*' * len(word)
        for word in text.split())

    forbin = []
    with open("forbiden_words.txt") as dfcensor:
        for line in dfcensor:
            forbin.append(str(line).replace("\n", ""))

    # The last time a recording was retreived from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    source = sr.Microphone(sample_rate=16000)

    # Which model to use
    model = args.model
    if args.model != "large-v2" and not args.non_english:
        model = model + ".en"

    # Choose cpu or gpu
    device = args.device
    if device == "cpu":
        compute_type = "int8"
    else:
        compute_type = args.compute_type
    cpu_threads = args.threads

    nltk.download('punkt_tab')
    audio_model = WhisperModel(model, device = device, compute_type = compute_type , cpu_threads = cpu_threads)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name 
    transcription = ['']

    # This only has to run once
    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print("Model loaded.\n")

    while True:
        try:
            now = datetime.now(UTC)
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not data_queue.empty():
                    data = data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                text = ""
                    
                segments, info = audio_model.transcribe(temp_file)
                for segment in segments:
                    text += segment.text
                #text = result['text'].strip()

                # If we detected a pause between recordings, add a new item to our transcripion.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text
                last_four_elements = transcription[-10:]
                result = ''.join(last_four_elements)    
                sentences = sent_tokenize(result)
                # Send out text here
                # Clear the console to reprint the updated transcription.

                # Infinite loops are bad for processors, must sleep.
                #print("\n\nTranscription:")
                last = ""
                for line in transcription:
                    if last == line:
                        print("Duplicate lines detected.")
                    else:
                        last = line

                        print(f"-----\n\n{line}")
                        with open("output.txt", "w") as file:
                            output_str = ""
                            for item in transcription:
                                output_str = output_str + f"{item}\n"
                            file.write(f"\n{output_str}")


        except KeyboardInterrupt:
            break

    # print("\n\nTranscription:")
    # for line in transcription:
    #     print(line)

if __name__ == "__main__":
    
    main()