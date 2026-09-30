# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "openai",
#     "pynput",
#     "pyperclip",
#     "pyyaml",
#     "requests",
#     "sounddevice",
# ]
# ///
import io, wave, base64, requests
import numpy as np
import sounddevice as sd
import pyperclip
from pynput import keyboard

import yaml

with open('simple.conf') as conffile:
    conf = yaml.safe_load(conffile)

# API setup - prioritize Gemini if key exists (and isn't a placeholder)
GEMINI_API_KEY = conf.get('gemini', {}).get('api_key', '')
USE_GEMINI = GEMINI_API_KEY and not GEMINI_API_KEY.startswith('YOUR_')

if USE_GEMINI:
    GEMINI_MODEL = conf['gemini'].get('model', 'gemini-3.1-flash-lite')
    print(f"Using Google Gemini ({GEMINI_MODEL}) for transcription")
else:
    from openai import OpenAI
    client = OpenAI(api_key = conf['openai']['api_key'])
    print("Using OpenAI Whisper for transcription")

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
CHANNELS = 1
CHUNK = 1024

audio_device_index = conf['audio']['device']
print("\n\n----------------------Recording device list---------------------")
for i, device_info in enumerate(sd.query_devices()):
  if device_info.get('max_input_channels')>0:
    print(f"{'**' if audio_device_index==i else '  '} Input Device id {i} "+
          f" {device_info.get('name')} "+
          f" @{device_info.get('default_samplerate')}Hz")
print("-------------------------------------------------------------")

SAMPLE_RATE = int( sd.query_devices(audio_device_index).get('default_samplerate') )

# Globals to manage ongoing recordings
frames = []  # A buffer to store audio chunks
stream, is_recording = None, False  # Flag to manage recording state

# 3 Functions to handle audio recording
def start_recording():
  global is_recording, frames, stream
  frames = []
  is_recording = True
  stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                          dtype='int16', blocksize=CHUNK,
                          device=audio_device_index,
                          callback=_get_callback(),
                          )
  stream.start()
  print("Recording started...")

def _get_callback():
  def callback(indata, frame_count, time_info, status):
    global frames
    frames.append(indata.copy())
    print(f"Continuing Recording... {len(frames)=}")
  return callback

def stop_recording(action):
  global is_recording, stream
  is_recording = False
  print(f"Recording finished : {len(frames)=}")
  stream.stop()
  stream.close()
  save_and_transcribe_audio(action)

def save_and_transcribe_audio(action):
  buffer = io.BytesIO()  # Using a buffer requires no temporary on-disk file
  buffer.name = "buffer.wav"
  audio_data = np.concatenate(frames, axis=0) if frames else np.zeros((0, CHANNELS), dtype='int16')
  wf = wave.open(buffer, 'wb')
  wf.setnchannels(CHANNELS)
  wf.setsampwidth(2)  # int16 = 2 bytes
  wf.setframerate(SAMPLE_RATE)
  wf.writeframes(audio_data.tobytes())
  wf.close()

  if USE_GEMINI:
    show_notification("Transcribe-to-Clipboard", "Sending to Gemini API")
    print(f"Sending buffer to Gemini API")
    transcript = transcribe_with_gemini(buffer)
  else:
    show_notification("Transcribe-to-Clipboard", "Sending to OpenAI whisper API")
    print(f"Sending buffer to OpenAI whisper API")
    # https://platform.openai.com/docs/api-reference/audio/createTranscription
    transcript = client.audio.transcriptions.create(
      model='whisper-1',
      file=buffer,
      response_format='text'
    )
  transcript = transcript.strip()
  print("Transcription: ", transcript)
  if action == 'type':
    type_text(transcript, HOTKEYS[action])
    show_notification("Transcribe-to-Clipboard", f"Typed: {transcript[:100]}...")
  else:
    pyperclip.copy(transcript)
    show_notification("Transcribe-to-Clipboard", f"{transcript[:100]}...")

def transcribe_with_gemini(buffer):
  buffer.seek(0)
  audio_b64 = base64.standard_b64encode(buffer.read()).decode('utf-8')

  url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

  payload = {
    "contents": [{
      "parts": [
        {"text": "Transcribe this audio exactly. Output only the transcription, no commentary."},
        {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}}
      ]
    }]
  }

  response = requests.post(url, json=payload)
  response.raise_for_status()

  result = response.json()
  return result['candidates'][0]['content']['parts'][0]['text']


import time

keyboard_controller = keyboard.Controller()

def type_text(text, combo):
  # The key that ended the combo has just been released, but held modifiers
  # (cmd/alt) may still be physically down - wait for them to clear so the
  # synthetic keystrokes below aren't interpreted as more Alt/Cmd shortcuts.
  deadline = time.time() + 3.0
  while any(k in current_keys for k in combo) and time.time() < deadline:
    time.sleep(0.02)
  keyboard_controller.type(text)

# 2 Listeners for keyboard activity
#key_combo = [keyboard.Key.ctrl_l, keyboard.Key.alt_l, keyboard.KeyCode.from_char('w')]
# NB: Cannot use 's' or 'z' due to the effect of Ctrl-S and Ctrl-Z on terminal...
#key_combo = [keyboard.Key.cmd, keyboard.Key.tab]   # Just 'Windows-Tab' for Speech copy! # But AIstudio/Chrome hates it!
HOTKEYS = {
  'clipboard': [keyboard.Key.cmd, keyboard.Key.alt_l, keyboard.KeyCode.from_char('c')],  # Windows-Alt-c
  'type':      [keyboard.Key.cmd, keyboard.Key.alt_l, keyboard.KeyCode.from_char('v')],  # Windows-Alt-v
}

KEY_LABELS = {
  keyboard.Key.cmd: 'Windows', keyboard.Key.cmd_l: 'Windows', keyboard.Key.cmd_r: 'Windows',
  keyboard.Key.alt: 'Alt', keyboard.Key.alt_l: 'Alt', keyboard.Key.alt_r: 'Alt', keyboard.Key.alt_gr: 'AltGr',
  keyboard.Key.ctrl: 'Ctrl', keyboard.Key.ctrl_l: 'Ctrl', keyboard.Key.ctrl_r: 'Ctrl',
  keyboard.Key.shift: 'Shift', keyboard.Key.shift_l: 'Shift', keyboard.Key.shift_r: 'Shift',
}

def format_key(key):
  if key in KEY_LABELS:
    return KEY_LABELS[key]
  if isinstance(key, keyboard.KeyCode) and key.char:
    return key.char
  return str(key)

def format_combo(combo):
  return '-'.join(format_key(k) for k in combo)

print("\n\nActions:")
print(f"* Press-to-Talk {format_combo(HOTKEYS['clipboard'])} - Release to copy transcript to clipboard")
print(f"* Press-to-Talk {format_combo(HOTKEYS['type'])} - Release to type transcript at the cursor")
print("* Ctrl-c to exit")

active_action = None

def on_press(key):
  global is_recording, active_action
  #print(f"on_press({key=}, {current_keys=}")
  if is_recording:
    return
  for action, combo in HOTKEYS.items():
    if all(k in current_keys for k in combo):
      print(f"All hotkeys pressed : Start recording! ({action=})")
      active_action = action
      start_recording()
      break

def on_release(key):
  global is_recording, active_action
  #print(f"on_release({key=}, new {current_keys=}")
  if is_recording and active_action is not None and key in HOTKEYS[active_action]:
    print("Released something relevant : Stop recording!")
    stop_recording(active_action)  # This processes the audio
    active_action = None

#   Maintain a set of current keys pressed
current_keys = set()
with keyboard.Listener(
    on_press=lambda key: current_keys.add(key) or on_press(key),
    on_release=lambda key: current_keys.discard(key) or on_release(key)
  ) as listener:
  listener.join()
