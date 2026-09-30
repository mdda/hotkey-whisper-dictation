# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "onnx-asr[cpu,hub]",
#     "openai",
#     "pillow",
#     "pynput",
#     "pyperclip",
#     "pystray",
#     "pyyaml",
#     "requests",
#     "soxr",
#     "sounddevice",
# ]
# ///
# Cloud transcription (Gemini, or OpenAI Whisper).  The local Silero VAD (see dictation_common.py)
#   lets dictation mode send each phrase as you pause.
import base64, requests

import dictation_common as dc

conf = dc.conf

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

CONTEXT_CHARS = 500  # How much of the preceding dictation to pass along with each phrase

def transcribe_with_gemini(audio, context):
  audio_b64 = base64.standard_b64encode(dc.to_wav(audio).read()).decode('utf-8')

  url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

  prompt = "Transcribe this audio exactly. Output only the transcription, no commentary."
  if context:
    prompt += ("\nThe audio continues a dictation. For context only (do not repeat it), the text so far is:\n"
               + context[-CONTEXT_CHARS:])
  payload = {
    "contents": [{
      "parts": [
        {"text": prompt},
        {"inline_data": {"mime_type": "audio/wav", "data": audio_b64}}
      ]
    }]
  }

  response = requests.post(url, json=payload, timeout=30)
  response.raise_for_status()

  result = response.json()
  return result['candidates'][0]['content']['parts'][0]['text']

def transcribe_with_openai(audio, context):
  # https://platform.openai.com/docs/api-reference/audio/createTranscription
  extra = {'prompt': context[-CONTEXT_CHARS:]} if context else {}
  return client.audio.transcriptions.create(
    model='whisper-1',
    file=dc.to_wav(audio),
    response_format='text',
    **extra,
  )

# Clipboard (hold) sends the whole recording on release, so the model sees the full context.
#   Dictation (toggle) sends each phrase as you pause, so text appears as you go.
dc.run('Transcribe-via-API', transcribe_with_gemini if USE_GEMINI else transcribe_with_openai,
       whole_clip_actions=['clipboard'])
