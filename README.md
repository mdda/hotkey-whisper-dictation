# Linux Transcription on hotkey
##  First backend : Whisper (via OpenAI API)


### Installation (Fedora)

This project uses [`uv`](https://docs.astral.sh/uv/) with dependencies declared inline in
`transcribe-via-api.py` and `transcribe-local.py` (PEP 723), so there's no virtual environment to create or
activate by hand - `uv run` resolves and caches everything automatically the first time
it's run.

```bash
sudo dnf install gcc python3-devel xclip
```

(`gcc` + `python3-devel` are needed once, to build `evdev`, a dependency of the `pynput`
keyboard-monitoring library. `xclip` is needed at runtime for clipboard access. Audio
capture uses `sounddevice`, which only needs the `portaudio` runtime library - already
present on most Fedora installs.)

### Running

Copy `TEMPLATE_simple.conf` to `simple.conf`, and update with your OpenAI or Gemini API key.

Run the following:
```bash
./start-listening-gemini
```
(equivalent to `uv run transcribe-via-api.py`, but can be invoked from any directory -
handy for autostart entries.)

If the index of the audio device for recording isn't correct, put the new index value
from the list of devices into your `simple.conf` and restart 

There are two hotkeys:
* `Windows-Alt-c` - "Press-and-Hold" walkie-talkie : transcribes and copies the result
  to the clipboard when released.  The whole recording is sent in one go, so the model
  sees the full context.
* `Windows-Alt-d` - dictation mode **toggle** : press once to start, and each phrase is
  typed at the current cursor position (via simulated keystrokes - no clipboard involved)
  as you pause; press again to stop.  (Toggle rather than hold, since typing can't
  happen while Windows-Alt are physically held down.)

A microphone icon in the system tray (e.g. XFCE's Status Tray plugin) shows the state :
grey = idle, red = listening, amber = finishing the last phrase.  Its menu has a `Quit` item.

Pauses are detected by a small local voice-activity model (Silero VAD, ~2MB, ~0.4% of
one CPU core while listening).  In dictation mode each phrase is sent along with the
text dictated so far, for context.

These can be changed by looking for `HOTKEYS = {}` in `dictation_common.py`, which
holds everything shared between the cloud and local versions.

The `transcribe-via-api.py` program can just be left running in the background - 
it's light-weight, and the only data ever to get sent up to OpenAI is the audio
recorded while listening.



##  Alternative backend : Parakeet (fully local, CPU)

`transcribe-local.py` does the transcription on-device with no API key, using
NVIDIA's Parakeet-TDT 0.6B (int8 ONNX, via [`onnx-asr`](https://github.com/istupakov/onnx-asr)).
Silero VAD splits your speech into phrases at each pause, and each phrase is
transcribed as soon as it ends:

```bash
./start-listening-local
```
(equivalent to `uv run transcribe-local.py`, runnable from any directory.)

The first run downloads ~650MB of model files into `~/.cache/huggingface`. Once loaded,
the process uses ~1.2GB of RAM, and a 5s phrase takes ~0.5s to transcribe on a
4-core laptop CPU.

The hotkeys and tray icon are the same as above - but here, since local transcription
is fast, `Windows-Alt-c` also transcribes phrase-by-phrase while you're talking, so only
the last phrase needs processing once the keys are released.

Optional settings in `simple.conf`:
```yaml
local:
  model: parakeet-redux  # Or nemo-parakeet-tdt-0.6b-v3 (multilingual); default is the English-only ...-v2
vad:
  min_silence_ms: 500    # Pause length that ends a phrase (both versions)
```

`parakeet-redux` is [moondream's 1.58-bit Parakeet-v3](https://huggingface.co/moondream/parakeet-redux)
(via [this ONNX export](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-redux-onnx)) :
a ~210MB download, ~650MB RAM, and slightly faster - but slightly less accurate than
the full models, particularly with background noise.
