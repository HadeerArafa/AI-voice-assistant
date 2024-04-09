import pyaudio
import numpy as np
import wave
import speech_recognition as sr
from matplotlib import pyplot as plt
import threading
import queue

# Configuration for audio stream
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SILENCE_THRESHOLD = 1000
SILENCE_DURATION = 3

# Initialize PyAudio and queue for data exchange
p = pyaudio.PyAudio()
audio_queue = queue.Queue()

def audio_streaming(queue):
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("* recording")
    frames = []
    silence_frames = 0
    recording = True
    
    while recording:
        data = stream.read(CHUNK)
        frames.append(data)
        audio_data = np.frombuffer(data, dtype=np.int16)
        max_amplitude = np.max(np.abs(audio_data))
        queue.put(audio_data)  # Put audio data into queue
        
        if max_amplitude < SILENCE_THRESHOLD:
            silence_frames += 1
            if silence_frames >= SILENCE_DURATION * (RATE / CHUNK):
                recording = False
        else:
            silence_frames = 0
    
    stream.stop_stream()
    stream.close()
    queue.put(None)  # Signal that recording is done
    print("* done recording")
    
    # Save the recorded data as a WAV file
    output_filename = "recorded_audio.wav"
    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Audio recorded and saved as {output_filename}")

def plot_audio(queue):
    plt.ion()
    fig, ax = plt.subplots()
    x = np.arange(0, 2 * CHUNK, 2)
    line, = ax.plot(x, np.zeros(CHUNK))
    ax.set_ylim(-2**15, 2**15)
    
    while True:
        data = queue.get()
        if data is None:  # Check if recording is done
            break
        line.set_ydata(data)
        plt.pause(0.001)

# Start audio streaming and plotting in separate threads
stream_thread = threading.Thread(target=audio_streaming, args=(audio_queue,))
plot_thread = threading.Thread(target=plot_audio, args=(audio_queue,))

stream_thread.start()
plot_thread.start()

stream_thread.join()
plot_thread.join()

# Proceed with speech recognition using the saved audio file
recognizer = sr.Recognizer()
audio_file_path = 'recorded_audio.wav'
with sr.AudioFile(audio_file_path) as source:
    audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        print("Recognized text:", text)
    except sr.UnknownValueError:
        print("Google Web Speech API could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")

plt.ioff()
plt.show()

# Cleanup
p.terminate()
