import openwakeword
import os
import pvporcupine
import pyaudio
import queue
import struct
import yaml
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from openai import OpenAI
from openwakeword.model import Model
from playsound import playsound

SCRIPT_PATH=os.path.dirname(os.path.realpath(__file__))
MODELS_PATH=os.path.join(SCRIPT_PATH, 'models')
SOUNDS_PATH=os.path.join(SCRIPT_PATH, 'sounds')

BASE_PROMPT=[
    { 'role': 'system',
      'content': 'Your name is Buddy. You are friendly, enthusiastic, and offer concise responses appropriate for young children.'},
]

with open(os.path.join(SCRIPT_PATH, 'config.yml'), 'r') as file:
    CONFIG = yaml.safe_load(file)

gpt_client = OpenAI(api_key=CONFIG['keys']['openai'])

def wake_word():
    return wake_word_pa_oww()

def wake_word_pa_oww():
    owwModel = Model(
        wakeword_models=[os.path.join(MODELS_PATH, CONFIG['models']['openwakeword'])]
    )
    frame_size = 2048
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=16000,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=frame_size)

    while True:
        frame = np.frombuffer(audio_stream.read(frame_size), dtype=np.int16)
        prediction = owwModel.predict(frame)

        if prediction['hey-buddy'] >= 0.5:
            print(prediction)
            audio_stream.close()
            return True

def wake_word_sd_porcupine():
    porcupine = pvporcupine.create(
        access_key=CONFIG['keys']['picovoice'],
        keywords=['hey buddy'],
        keyword_paths=[os.path.join(MODELS_PATH, CONFIG['models']['pv_porcupine'])]
    )

    frames = queue.Queue()
    with sd.InputStream(samplerate=porcupine.sample_rate,
                        blocksize=porcupine.frame_length,
                        channels=1,
                        dtype='int16',
                        callback=lambda indata, *_: frames.put(indata.copy())):
        while True:
            frame = struct.unpack_from('h' * porcupine.frame_length, frames.get())
            result = porcupine.process(frame)

            if result == 0:
                porcupine.delete()
                return True

def wake_word_pa_porcupine():
    porcupine = pvporcupine.create(
        access_key=CONFIG['keys']['picovoice'],
        keywords=['hey buddy'],
        keyword_paths=[os.path.join(MODELS_PATH, CONFIG['models']['pv_porcupine'])]
    )
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        frames_per_buffer=porcupine.frame_length,
        format=pyaudio.paInt16,
        channels=1,
        input=True)

    while True:
        frame = struct.unpack_from('h' * porcupine.frame_length, audio_stream.read(porcupine.frame_length))
        result = porcupine.process(frame)

        if result == 0:
            audio_stream.close()
            porcupine.delete()
            return True

def record_prompt():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            print("==> Listening for prompt...")
            audio = r.listen(source, timeout=10, phrase_time_limit=30)
            print("==> Finished recording prompt\n")
            return r.recognize_google_cloud(audio)
        except sr.WaitTimeoutError:
            print("[ERROR]: Timed out waiting to record prompt")
        except sr.UnknownValueError:
            print("[ERROR]: Recording was not understood")
        except sr.RequestError as e:
            print(f"[ERROR]: Could not get results from speech-to-text service: {e}")

    return None

def ask_chatgpt(prompt):
    completion = gpt_client.chat.completions.create(
        model='gpt-4o',
        messages=BASE_PROMPT + [{'role': 'user', 'content': prompt}]
    )
    return completion.choices[0].message.content

def play_response(text):
    with gpt_client.audio.speech.with_streaming_response.create(
            model='tts-1',
            voice='nova',
            input=text,
    ) as response:
        response.stream_to_file(os.path.join(SOUNDS_PATH, 'response.mp3'))

    print("==> Done. Playing response audio...")
    playsound(os.path.join(SOUNDS_PATH, 'response.mp3'))
    print("==> Finished playing response audio\n")

def main():
    new_session = True

    while True:
        if new_session:
            print("==> Listening for wake word...")
            if wake_word():
                print("==> Wake word detected!\n")
                new_session = False
                playsound(os.path.join(SOUNDS_PATH, 'awake.wav'))

        print("==> Start speaking...")
        prompt = record_prompt()
        playsound(os.path.join(SOUNDS_PATH, 'done.wav'))
        if prompt is None:
            print("==> Prompt could not be recorded")
            play_response("I didn't get that. Please try again later.")
            new_session = True
            continue
        elif prompt.strip().lower() == 'goodbye':
            print("==> 'Goodbye' detected. Ending session...")
            play_response('Goodbye!')
            new_session = True
            continue
        else:
            print(f"==> Recorded prompt: {prompt}\n")

        # Send text to ChatGPT.
        print(f"==> Sending prompt to ChatGPT...")
        response = ask_chatgpt(prompt)
        print(f"==> Response: {response}\n")

        print("==> Converting text to audio...")
        play_response(response)
        print("==> Finished converting audio\n")


if __name__ == '__main__':
    main()

