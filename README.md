# Linux Transcription on hotkey
##  First backend : Whisper (via OpenAI API)


### Installation (Fedora)

This project uses [`uv`](https://docs.astral.sh/uv/) with dependencies declared inline in
`transcribe-to-clipboard.py` (PEP 723), so there's no virtual environment to create or
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
./start-listening
```
(equivalent to `uv run transcribe-to-clipboard.py`, but can be invoked from any directory -
handy for autostart entries.)

If the index of the audio device for recording isn't correct, put the new index value
from the list of devices into your `simple.conf` and restart 

There are two "Press-and-Hold" walkie-talkie hotkeys:
* `Windows-Alt-c` - transcribes and copies the result to the clipboard.
* `Windows-Alt-v` - transcribes and types the result directly at the current cursor
  position (via simulated keystrokes - no clipboard involved).

These can be changed by looking for `HOTKEYS = {}` in the code.

The `transcribe-to-clipboard.py` program can just be left running in the background - 
it's light-weight, and the only data ever to get sent up to OpenAI is the audio
recorded while the hotkey combo is being pressed.

