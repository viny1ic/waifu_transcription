#!/usr/bin/env python3
import os
import json
import requests
from pathlib import Path
from zipfile import ZipFile

import pyaudio
from vosk import Model, KaldiRecognizer

# ——— Configuration ———
MODEL_URL    = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
MODEL_ZIP    = Path("vosk-model-small-en-us-0.15.zip")
MODEL_DIR    = Path("model")  # final folder name will be ./model

AUDIO_RATE   = 16000
CHUNK_SIZE   = 8000
LISTEN_TIME  = 5.0  # seconds

# ——— Download & extract model if needed ———
def ensure_model():
    if MODEL_DIR.exists():
        print(f"✔ Model already present at '{MODEL_DIR}/'")
        return

    print(f"⟳ Downloading model from {MODEL_URL} …")
    resp = requests.get(MODEL_URL, stream=True)
    resp.raise_for_status()
    with open(MODEL_ZIP, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)

    print("⟳ Extracting model…")
    with ZipFile(MODEL_ZIP, "r") as zipf:
        zipf.extractall()
    # Rename extracted folder to a consistent name
    extracted = Path("vosk-model-small-en-us-0.15")
    extracted.rename(MODEL_DIR)
    MODEL_ZIP.unlink()  # remove the zip to save space
    print(f"✔ Model is ready at '{MODEL_DIR}/'")

# ——— Capture & transcribe ———
def listen_and_transcribe(duration: float = LISTEN_TIME) -> str:
    # Load offline model
    model = Model(str(MODEL_DIR))
    rec   = KaldiRecognizer(model, AUDIO_RATE)

    # Open microphone stream
    pa     = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=AUDIO_RATE,
                     input=True,
                     frames_per_buffer=CHUNK_SIZE)
    stream.start_stream()

    print(f"🎤 Listening for {duration} seconds…")
    chunks = int(AUDIO_RATE / CHUNK_SIZE * duration)
    texts  = []

    for _ in range(chunks):
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        if rec.AcceptWaveform(data):
            part = json.loads(rec.Result()).get("text", "")
            if part:
                texts.append(part)

    # Final partial result
    final = json.loads(rec.FinalResult()).get("text", "")
    if final:
        texts.append(final)

    # Cleanup
    stream.stop_stream()
    stream.close()
    pa.terminate()

    return " ".join(texts)

# ——— Main entrypoint ———
if __name__ == "__main__":
    # Dependencies:
    #   pip install vosk requests pyaudio
    ensure_model()
    transcript = listen_and_transcribe()
    print("🗣️ You said:", transcript)
