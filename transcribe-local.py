# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "onnx-asr[cpu,hub]",
#     "pillow",
#     "pynput",
#     "pyperclip",
#     "pystray",
#     "pyyaml",
#     "soxr",
#     "sounddevice",
# ]
# ///
# Fully-local dictation : Silero VAD splits the audio into phrases as you speak,
#   and each phrase is transcribed by Parakeet-TDT (ONNX) on the CPU.
import time
import onnx_asr

import dictation_common as dc

local_conf = dc.conf.get('local', {}) or {}
# 'nemo-parakeet-tdt-0.6b-v2' (English), 'nemo-parakeet-tdt-0.6b-v3' (25 languages) or 'parakeet-redux'
ASR_MODEL = local_conf.get('model', 'nemo-parakeet-tdt-0.6b-v2')

def load_redux():
  # moondream's 1.58-bit (ternary) Parakeet-v3, as exported to onnx-asr layout by Olicorne.
  #   The weights are ternary, so the 2-bit encoder is lossless (190MB vs 643MB for int8)
  #   onnx-asr only knows the quantization-suffix naming scheme, so link the chosen files
  #   into a local directory under the plain names it expects.
  import os
  from pathlib import Path
  from huggingface_hub import hf_hub_download
  repo = 'Olicorne/parakeet-tdt-0.6b-v3-redux-onnx'
  model_dir = Path(os.path.expanduser('~/.cache/hotkey-dictation/parakeet-redux'))
  model_dir.mkdir(parents=True, exist_ok=True)
  for src, dst in {'w2a8/encoder-model.w2a8.onnx':       'encoder-model.onnx',
                   'int8/decoder_joint-model.int8.onnx': 'decoder_joint-model.onnx',
                   'vocab.txt': 'vocab.txt', 'config.json': 'config.json'}.items():
    cached = Path(hf_hub_download(repo, src))
    link = model_dir / dst
    if link.is_symlink() and link.resolve() != cached.resolve():
      link.unlink()  # Repo updated : re-point to the new snapshot
    if not link.exists():
      link.symlink_to(cached)
  return onnx_asr.load_model('nemo-conformer-tdt', path=model_dir)

t0 = time.time()
if ASR_MODEL == 'parakeet-redux':
  print("Loading parakeet-redux (2-bit) - first run downloads ~210MB from Hugging Face...")
  asr = load_redux()
else:
  print(f"Loading {ASR_MODEL} (int8) - first run downloads ~650MB from Hugging Face...")
  asr = onnx_asr.load_model(ASR_MODEL, quantization='int8')
print(f"Model loaded in {time.time()-t0:.1f}s")

def transcribe(audio, context):
  return asr.recognize(audio, sample_rate=dc.ASR_RATE)  # Parakeet has no use for the context

# Local transcription is fast, so both hotkeys transcribe phrase-by-phrase as you speak
dc.run('Transcribe-Local', transcribe)
