#!/usr/bin/env python3
import sys
import json
import threading
import requests
from pathlib import Path
from zipfile import ZipFile

import pyaudio
from vosk import Model, KaldiRecognizer

# ——— Load configuration from JSON ———
with open("models.json", "r", encoding="utf-8") as f:
    raw_cfg = json.load(f)

LANG_CONFIG = {
    key: {
        "name": v["name"],
        "dir": Path(v["dir"]),
        "zip": v["zip"],
        "url": v["url"]
    }
    for key, v in raw_cfg.items()
}

AUDIO_RATE = 16000
CHUNK_SIZE = 8000

# ——— Keypress helper ———
if sys.platform.startswith("win"):
    import msvcrt
    def wait_for_space(prompt: str):
        print(prompt)
        while True:
            if msvcrt.kbhit() and msvcrt.getch() == b' ':
                return
else:
    import tty, termios, select
    def wait_for_space(prompt: str):
        print(prompt)
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if r and sys.stdin.read(1) == ' ':
                    return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def ensure_model(lang_key):
    cfg = LANG_CONFIG[lang_key]
    target_dir = cfg["dir"]
    zip_path = Path(cfg["zip"])

    if target_dir.exists():
        return

    print(f"↳ Downloading {cfg['name']} model…")
    resp = requests.get(cfg["url"], stream=True)
    resp.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)

    print("↳ Extracting…")
    with ZipFile(zip_path, "r") as zipf:
        zipf.extractall()
    extracted = next(Path().glob("vosk-model-small-*"))
    extracted.rename(target_dir)
    zip_path.unlink()
    print(f"✔ {cfg['name']} model ready at {target_dir}/")


def record_until_space() -> bytes:
    wait_for_space("▶ Press [space] to START recording…")
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=AUDIO_RATE,
                     input=True,
                     frames_per_buffer=CHUNK_SIZE)
    print("● Recording… Press [space] to STOP.")
    frames = []
    stop_flag = threading.Event()

    def stopper():
        wait_for_space("")  # no prompt
        stop_flag.set()

    threading.Thread(target=stopper, daemon=True).start()
    while not stop_flag.is_set():
        frames.append(stream.read(CHUNK_SIZE, exception_on_overflow=False))

    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("■ Recording stopped.")
    return b"".join(frames)


def debug_language_scores(results: dict):
    art = r"""
       (\_/)
      ( •_•)  “Debug Mode Activated!”
     <)   )\   Here’s what each model heard:
      /   \ 
    """
    print(art)
    for key, data in results.items():
        name = LANG_CONFIG[key]["name"]
        text = data["text"] or "[no words]"
        count = data["count"]
        print(f"   • {name:<7} : {count:>2} words  → “{text}”")
    print("\n(＾◡＾)╯ Language scoring complete!\n")


def detect_language(sample: bytes) -> str:
    # run each model and collect its raw text + count
    results = {}
    for key, cfg in LANG_CONFIG.items():
        model = Model(str(cfg["dir"]))
        rec = KaldiRecognizer(model, AUDIO_RATE)
        rec.AcceptWaveform(sample)
        text = json.loads(rec.FinalResult()).get("text", "")
        results[key] = {
            "text": text,
            "count": len(text.split())
        }
    debug_language_scores(results)
    # choose the language with the highest word count
    return max(results, key=lambda k: results[k]["count"])


def transcribe(sample: bytes, lang_key: str) -> str:
    cfg = LANG_CONFIG[lang_key]
    model = Model(str(cfg["dir"]))
    rec = KaldiRecognizer(model, AUDIO_RATE)
    rec.AcceptWaveform(sample)
    return json.loads(rec.FinalResult()).get("text", "")


if __name__ == "__main__":
    # 1) Ensure all models are downloaded
    for lang in LANG_CONFIG:
        ensure_model(lang)

    # 2) Record until spacebar pressed twice
    audio_data = record_until_space()

    # 3) Detect language (with debug output)
    chosen = detect_language(audio_data)
    print(f"🎯 Detected language: {LANG_CONFIG[chosen]['name']}\n")

    # 4) Transcribe and show result
    transcript = transcribe(audio_data, chosen)
    print("📝 Final Transcription:", transcript)
