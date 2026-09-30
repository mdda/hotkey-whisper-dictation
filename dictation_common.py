# Shared machinery for transcribe-local.py and transcribe-via-api.py :
#   audio capture, Silero VAD phrase splitting, hotkeys, typing/clipboard output and a tray icon.
#   Each script supplies a transcribe(audio, context) function and calls run().
#   (uv runs each script with only its own PEP 723 dependencies, so both scripts list this module's too.)
import io, os, queue, signal, subprocess, threading, time, wave
os.environ.setdefault('PYSTRAY_BACKEND', 'xorg')  # Plain XEmbed tray icon : works in the XFCE Status Tray without GTK/PyGObject
import numpy as np
import sounddevice as sd
import soxr
import pyperclip
from pynput import keyboard

import yaml

with open('simple.conf') as conffile:
  conf = yaml.safe_load(conffile)

MIN_SILENCE_MS = (conf.get('vad', {}) or {}).get('min_silence_ms', 500)  # Pause length that ends a phrase


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
ASR_RATE = 16000  # Silero wants 16kHz, and it's what speech models use anyway

audio_device_index = conf['audio']['device']
SAMPLE_RATE = int( sd.query_devices(audio_device_index).get('default_samplerate') )

def print_devices():
  print("\n\n----------------------Recording device list---------------------")
  for i, device_info in enumerate(sd.query_devices()):
    if device_info.get('max_input_channels')>0:
      print(f"{'**' if audio_device_index==i else '  '} Input Device id {i} "+
            f" {device_info.get('name')} "+
            f" @{device_info.get('default_samplerate')}Hz")
  print("-------------------------------------------------------------")

def to_wav(audio):
  """16kHz float32 audio -> in-memory WAV file (for APIs that want an upload)."""
  buffer = io.BytesIO()  # Using a buffer requires no temporary on-disk file
  buffer.name = "buffer.wav"
  wf = wave.open(buffer, 'wb')
  wf.setnchannels(CHANNELS)
  wf.setsampwidth(2)  # int16 = 2 bytes
  wf.setframerate(ASR_RATE)
  wf.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
  wf.close()
  buffer.seek(0)
  return buffer


def load_vad():
  import onnx_asr
  import onnxruntime as rt
  vad_options = rt.SessionOptions()  # Silero is tiny : one thread is faster than the default thread pool, at ~1/5 the CPU
  vad_options.intra_op_num_threads = vad_options.inter_op_num_threads = 1
  return onnx_asr.load_vad('silero', sess_options=vad_options)


# Streaming Silero VAD : fed 512-sample hops (32ms @ 16kHz), emits finished phrases
class PhraseSegmenter:
  HOP, CONTEXT = 512, 64
  THRESHOLD, NEG_THRESHOLD = 0.5, 0.35
  PRE_ROLL_HOPS = 10      # ~320ms of audio kept from before speech onset
  MIN_SPEECH_HOPS = 8     # ~250ms : shorter blips are ignored
  MAX_PHRASE_S = 20       # Force a cut in very long run-on speech

  def __init__(self, vad):
    self.session = vad._model  # onnx-asr's Silero only exposes whole-file segmentation, so drive its ONNX session directly
    self.min_silence_hops = int(MIN_SILENCE_MS / 1000 * ASR_RATE / self.HOP)
    self.reset()

  def reset(self):
    self.state = np.zeros((2, 1, 128), dtype=np.float32)
    self.context = np.zeros(self.CONTEXT, dtype=np.float32)
    self.pending = np.zeros(0, dtype=np.float32)
    self.hops = []           # Hops of the current phrase (including pre-roll)
    self.in_speech = False
    self.speech_hops = 0
    self.silent_hops = 0

  def _prob(self, hop):
    frame = np.concatenate([self.context, hop])[None, :]
    self.context = hop[-self.CONTEXT:]
    out, self.state = self.session.run(['output', 'stateN'],
                                       {'input': frame, 'state': self.state, 'sr': np.array([ASR_RATE], dtype=np.int64)})
    return float(out[0, 0])

  def feed(self, audio):
    """Add 16kHz float32 audio, yield any phrases that have just finished."""
    self.pending = np.concatenate([self.pending, audio])
    while len(self.pending) >= self.HOP:
      hop, self.pending = self.pending[:self.HOP], self.pending[self.HOP:]
      p = self._prob(hop)
      self.hops.append(hop)
      if not self.in_speech:
        if p >= self.THRESHOLD:
          self.in_speech, self.speech_hops, self.silent_hops = True, 1, 0
        else:
          self.hops = self.hops[-self.PRE_ROLL_HOPS:]
        continue
      if p < self.NEG_THRESHOLD:
        self.silent_hops += 1
      else:
        self.speech_hops += 1
        self.silent_hops = 0
      too_long = len(self.hops) * self.HOP >= self.MAX_PHRASE_S * ASR_RATE
      if self.silent_hops >= self.min_silence_hops or too_long:
        if (phrase := self._take_phrase()) is not None:
          yield phrase

  def flush(self):
    """Called when recording stops : return whatever speech is still buffered."""
    self.hops.append(self.pending)
    phrase = self._take_phrase() if self.in_speech else None
    self.reset()
    return phrase

  def _take_phrase(self):
    # Trim most of the trailing silence, but keep a little so the last word isn't clipped
    keep = len(self.hops) - max(self.silent_hops - 3, 0)
    phrase = np.concatenate(self.hops[:keep]) if self.speech_hops >= self.MIN_SPEECH_HOPS else None
    self.hops, self.in_speech, self.speech_hops, self.silent_hops = [], False, 0, 0
    return phrase


# Tray icon showing the current state : grey = idle, red = listening, amber = finishing the last phrase
tray = None

def make_icon(fill, outline):
  from PIL import Image, ImageDraw
  img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
  draw = ImageDraw.Draw(img)
  draw.rounded_rectangle((22, 6, 42, 38), radius=10, fill=fill, outline=outline, width=3)  # Microphone head
  draw.arc((14, 18, 50, 48), start=0, end=180, fill=outline, width=4)                     # Cradle
  draw.line((32, 48, 32, 58), fill=outline, width=4)
  draw.line((22, 58, 42, 58), fill=outline, width=4)
  return img

def make_tray(name):
  import pystray
  icons = {
    'idle':      make_icon((0, 0, 0, 0), (150, 150, 150, 255)),
    'listening': make_icon((230, 30, 30, 255), (230, 30, 30, 255)),
    'finishing': make_icon((240, 170, 0, 255), (240, 170, 0, 255)),
  }
  icon = pystray.Icon(name, icons['idle'], 'Dictation : idle',
                      menu=pystray.Menu(pystray.MenuItem('Quit', lambda icon, item: icon.stop())))
  icon.state_images = icons
  return icon

def set_status(state, detail=''):
  if tray is None:
    return
  tray.icon = tray.state_images[state]
  tray.title = f"Dictation : {state}{' ('+detail+')' if detail else ''}"


# Pipeline : audio callback -> audio_q -> VAD thread -> phrase_q -> ASR thread -> type/clipboard
#   Each recording is bracketed by ('start', action) and END_OF_RECORDING markers in both queues.
#   For actions in whole_clip_actions the VAD only checks there was speech, and the entire
#   recording is transcribed in one go on release (gives cloud models the full context).
audio_q, phrase_q = queue.Queue(), queue.Queue()
END_OF_RECORDING = None

def vad_worker(vad, whole_clip_actions):
  segmenter = PhraseSegmenter(vad)
  resampler, whole, clip, had_speech = None, False, [], False
  while True:
    item = audio_q.get()
    if isinstance(item, tuple):  # ('start', action)
      resampler = soxr.ResampleStream(SAMPLE_RATE, ASR_RATE, CHANNELS, dtype='float32')
      whole, clip, had_speech = item[1] in whole_clip_actions, [], False
      phrase_q.put(item)
      continue
    last = item is END_OF_RECORDING
    audio = resampler.resample_chunk(np.zeros(0, dtype=np.float32) if last else item, last=last)
    if whole:
      clip.append(audio)
    phrases = list(segmenter.feed(audio))
    if last and (phrase := segmenter.flush()) is not None:
      phrases.append(phrase)
    for phrase in phrases:
      had_speech = True
      if not whole:
        print(f"Phrase detected : {len(phrase)/ASR_RATE:.1f}s")
        phrase_q.put(phrase)
    if last:
      if whole and had_speech:
        phrase_q.put(np.concatenate(clip))
      phrase_q.put(END_OF_RECORDING)

def asr_worker(transcribe, name):
  action, transcript = None, []
  while True:
    phrase = phrase_q.get()
    if isinstance(phrase, tuple):  # ('start', action)
      action, transcript = phrase[1], []
      continue
    if phrase is END_OF_RECORDING:
      text = ' '.join(transcript)
      print("Transcription: ", text)
      if text and action == 'clipboard':
        pyperclip.copy(text)
        show_notification(name, f"{text[:100]}...")
      if not is_recording:
        set_status('idle')
      continue
    t0 = time.time()
    try:
      text = transcribe(phrase, ' '.join(transcript)).strip()
    except Exception as e:  # e.g. network trouble : keep the worker alive for the next phrase
      print(f"Transcription failed : {e!r}")
      show_notification(name, f"Transcription failed : {e}")
      continue
    print(f"  [{len(phrase)/ASR_RATE:.1f}s audio -> {time.time()-t0:.2f}s] {text!r}")
    if not text:
      continue
    if action == 'dictate':
      type_text((' ' if transcript else '') + text, HOTKEYS['dictate'])
    transcript.append(text)

def start_pipeline(transcribe, name, whole_clip_actions=()):
  vad = load_vad()
  threading.Thread(target=vad_worker, args=(vad, set(whole_clip_actions)), daemon=True).start()
  threading.Thread(target=asr_worker, args=(transcribe, name), daemon=True).start()


# Audio recording
stream, is_recording = None, False
recording_action = None

def start_recording(action):
  global is_recording, stream, recording_action
  recording_action = action
  is_recording = True
  audio_q.put(('start', action))
  stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                          dtype='float32', blocksize=CHUNK,
                          device=audio_device_index,
                          callback=lambda indata, frame_count, time_info, status: audio_q.put(indata[:, 0].copy()),
                          )
  stream.start()
  set_status('listening', action)
  print(f"Recording started... ({action=})")

def stop_recording():
  global is_recording, stream
  is_recording = False
  stream.stop()
  stream.close()
  audio_q.put(END_OF_RECORDING)
  set_status('finishing')
  print("Recording finished")


keyboard_controller = keyboard.Controller()

def type_text(text, combo):
  # Held modifiers (cmd/alt) would turn the synthetic keystrokes below into
  # Alt/Cmd shortcuts - wait for them to clear before typing.
  deadline = time.time() + 3.0
  while any(k in current_keys for k in combo) and time.time() < deadline:
    time.sleep(0.02)
  keyboard_controller.type(text)

# 2 Listeners for keyboard activity
# NB: Cannot use 's' or 'z' due to the effect of Ctrl-S and Ctrl-Z on terminal...
HOTKEYS = {
  'clipboard': [keyboard.Key.cmd, keyboard.Key.alt_l, keyboard.KeyCode.from_char('c')],  # Windows-Alt-c : hold
  'dictate':   [keyboard.Key.cmd, keyboard.Key.alt_l, keyboard.KeyCode.from_char('d')],  # Windows-Alt-d : toggle
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

# 'clipboard' is press-and-hold.  'dictate' is a toggle, since phrases can't be typed while
#   Windows-Alt are physically held.
def on_press(key):
  for action, combo in HOTKEYS.items():
    if key in combo and all(k in current_keys for k in combo):
      if not is_recording:
        start_recording(action)
      elif recording_action == 'dictate' and action == 'dictate':
        stop_recording()
      break

def on_release(key):
  if is_recording and recording_action == 'clipboard' and key in HOTKEYS['clipboard']:
    stop_recording()

#   Maintain a set of current keys pressed.  Auto-repeat re-sends presses for held keys,
#   so only a key's first press counts - otherwise holding the combo would toggle repeatedly.
current_keys = set()

def handle_press(key):
  if key in current_keys:
    return
  current_keys.add(key)
  on_press(key)

def handle_release(key):
  current_keys.discard(key)
  on_release(key)


def run(name, transcribe, whole_clip_actions=()):
  """transcribe(audio, context) : 16kHz float32 audio, plus the text so far in this recording -> text"""
  global tray
  print_devices()
  start_pipeline(transcribe, name, whole_clip_actions)
  tray = make_tray(name)

  print("\n\nActions:")
  print(f"* Press-to-Talk {format_combo(HOTKEYS['clipboard'])} - Release to copy transcript to clipboard")
  print(f"* Toggle {format_combo(HOTKEYS['dictate'])} - Dictation mode : press once to start, "
        "phrases are typed at the cursor as you pause, press again to stop")
  print("* The tray icon is red while listening")
  print("* Ctrl-c (or Quit from the tray icon menu) to exit")

  listener = keyboard.Listener(on_press=handle_press, on_release=handle_release)
  listener.start()
  signal.signal(signal.SIGINT, lambda signum, frame: tray.stop())
  tray.run()  # Blocks in the main thread until Quit / Ctrl-c
  listener.stop()
