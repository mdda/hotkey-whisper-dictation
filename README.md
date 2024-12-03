# Linux Transcription on hotkey
##  First backend : Whisper (via OpenAI API


### Installation (Fedora)

```bash
sudo dnf install portaudio-devel xclip
```

```bash
. ~/env312/bin/activate
uv pip install pyyaml   # Configuration file reading
uv pip install openai   # OpenAI API for Whisper
uv pip install pynput   # Keyboard monitoring
uv pip install pyaudio  # Audio reading
uv pip install pyperclip  # Clipboard interface
#uv pip install notify2 pydbus  # NOPE - notifications done via command line subprocess
```

### Running

Copy `TEMPLATE_simple.conf` to `simple.conf`, and update with your OpenAI API key.

Run the following:
```bash
. ~/env312/bin/activate
python transcribe-to-clipboard.py
```

If the index of the audio device for recording isn't correct, put the new index value
from the list of devices into your `simple.conf` and restart 

The current "Press-and-Hold" walkie-talkie hotkey combo is `Ctrl-Alt-w`.
This can be changed by looking for `key_combo=[]` in the code.

The `transcribe-to-clipboard.py` program can just be left running in the background - 
it's light-weight, and the only data ever to get sent up to OpenAI is the audio
recorded while the hotkey combo is being pressed.

