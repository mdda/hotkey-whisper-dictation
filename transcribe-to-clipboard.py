import io, wave
import pyaudio
import pyperclip
from pynput import keyboard

import yaml

with open('simple.conf') as conffile:
    conf = yaml.safe_load(conffile)

# OpenAI API key setup
from openai import OpenAI
client = OpenAI(api_key = conf['openai']['api_key'])

import subprocess
def show_notification(title, message):
  """Shows a desktop notification using notify-send."""
  try:
    subprocess.run(['notify-send', '--expire-time=1000', title, message], check=True)
  except FileNotFoundError:
    print("Error: notify-send command not found.  Ensure it's installed.")
  except subprocess.CalledProcessError as e:
    print(f"Error: notify-send failed with return code {e.returncode}")


# Setup audio stream parameters
FORMAT, CHANNELS = pyaudio.paInt16, 1
CHUNK = 1024

audio = pyaudio.PyAudio()

audio_device_index = conf['audio']['device']
print("\n\n----------------------Recording device list---------------------")
info = audio.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
  device_info = audio.get_device_info_by_host_api_device_index(0, i)
  #print(device_info.keys())
  if device_info.get('maxInputChannels')>0:
    print(f"{'**' if audio_device_index==i else '  '} Input Device id {i} "+
          f" {device_info.get('name')} "+
          f" @{device_info.get('defaultSampleRate')}Hz")
print("-------------------------------------------------------------")

audio_device_info = audio.get_device_info_by_host_api_device_index(0, audio_device_index)
SAMPLE_RATE = int( audio_device_info.get('defaultSampleRate') )

# Globals to manage ongoing recordings
frames = []  # A buffer to store audio chunks
stream, is_recording = None, False  # Flag to manage recording state

# 3 Functions to handle audio recording
def start_recording():
  global is_recording, frames, stream
  frames = []
  is_recording = True
  stream = audio.open(format=FORMAT, channels=CHANNELS,
                      rate=SAMPLE_RATE, input=True,
                      frames_per_buffer=CHUNK,
                      input_device_index=audio_device_index,
                      stream_callback = _get_callback(),
                      )
  print("Recording started...")

def _get_callback():
  def callback(in_data, frame_count, time_info, status):
    global frames # , stream
    frames.append(in_data)
    #print(f"Continuing Recording... {len(frames)=}")
    return in_data, pyaudio.paContinue
  return callback

def stop_recording():
  global is_recording, stream
  is_recording = False
  print(f"Recording finished : {len(frames)=}")
  stream.stop_stream()
  stream.close()
  save_and_transcribe_audio()

def save_and_transcribe_audio():
  buffer = io.BytesIO()  # Using a buffer requires no temporary on-disk file
  buffer.name = "buffer.wav"
  wf = wave.open(buffer, 'wb')
  wf.setnchannels(CHANNELS)
  wf.setsampwidth(audio.get_sample_size(FORMAT))
  wf.setframerate(SAMPLE_RATE)
  wf.writeframes(b''.join(frames))  # These are globally stored...
  wf.close()

  show_notification("Transcribe-to-Clipboard", "Sending to OpenAI whisper API")
  print(f"Sending buffer to OpenAI whisper API")
  # https://platform.openai.com/docs/api-reference/audio/createTranscription
  transcript = client.audio.transcriptions.create(
    model='whisper-1', 
    file=buffer, 
    response_format='text' 
  )  
  print("Transcription: ", transcript)
  pyperclip.copy(transcript)
  show_notification("Transcribe-to-Clipboard", f"{transcript[:100]}...")


# 2 Listeners for keyboard activity
key_combo = [keyboard.Key.ctrl_l, keyboard.Key.alt_l, keyboard.KeyCode.from_char('w')]

def on_press(key):
  global is_recording
  if all(k in current_keys for k in key_combo):
    if not is_recording:
      print("All hotkeys pressed : Start recording!")
      start_recording()

def on_release(key):
  global is_recording
  if key in key_combo:
    if is_recording:
      print("Released something relevant : Stop recording!")
      stop_recording()  # This processes the audio
      #return False  # Stop listener to exit program after 1 recording


# Setting up the listener for the keyboard
#   Maintain a set of current keys pressed
current_keys = set()
with keyboard.Listener(
    on_press=lambda key: current_keys.add(key) or on_press(key),
    on_release=lambda key: current_keys.discard(key) or on_release(key)
  ) as listener:
  listener.join()
