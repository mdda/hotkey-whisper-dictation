# Linux Transcription on hotkey
##  First backend : Whisper (via OpenAI API

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




