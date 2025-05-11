import sys
import threading
import random
import json
from pathlib import Path
from zipfile import ZipFile

import requests
import pyaudio
from tqdm import tqdm
from vosk import SetLogLevel, Model, KaldiRecognizer


class TranscriptionEngine:
    """
    A class-based offline multilingual transcription engine with optional waifu art,
    debug mode, and big-model toggle.
    """
    AUDIO_RATE = 16000
    CHUNK_SIZE = 8000

    def __init__(self, base_dir: Path, waifu: bool = False,
                 debug: bool = False, big: bool = False):
        # Base directory for assets
        self.base = base_dir
        self.waifu = waifu
        self.debug = debug
        self.variant = 'big' if big else 'small'

        # Paths
        self.arts_file = self.base / 'ascii_arts.txt'
        self.model_cfg = self.base / 'models.json'

        # Disable Vosk logs
        SetLogLevel(-1)

        # Load ASCII arts and model config
        self.arts = self._load_arts()
        raw_cfg = json.loads(self.model_cfg.read_text(encoding='utf-8'))
        self.lang_config = {}
        for key, v in raw_cfg.items():
            sel = v[self.variant]
            self.lang_config[key] = {
                'name': v['name'],
                'dir': self.base / sel['dir'],
                'zip': self.base / sel['zip'],
                'url': sel['url']
            }

    def _load_arts(self) -> list:
        text = self.arts_file.read_text(encoding='utf-8')
        return [block.strip() for block in text.split('\n===\n') if block.strip()]

    def _wait_for_space(self, prompt: str):
        print(prompt)
        if sys.platform.startswith('win'):
            import msvcrt
            while True:
                if msvcrt.kbhit() and msvcrt.getch() == b' ':
                    return
        else:
            import tty, termios, select
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

    def show_waifu(self):
        if self.waifu and self.arts:
            print(random.choice(self.arts), end='\n\n')

    def ensure_models(self):
        for lang_key in self.lang_config:
            cfg = self.lang_config[lang_key]
            target_dir = cfg['dir']
            zip_path = cfg['zip']
            if target_dir.exists():
                continue
            print(f"↳ Downloading {cfg['name']} ({self.variant}) model…")
            resp = requests.get(cfg['url'], stream=True)
            resp.raise_for_status()
            total = int(resp.headers.get('content-length', 0) or 0)
            with open(zip_path, 'wb') as f, tqdm(
                total=total, unit='iB', unit_scale=True,
                desc=f"{cfg['name']} 💖", ncols=60, ascii=True
            ) as bar:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
                    bar.update(len(chunk))
            print("↳ Extracting…")
            with ZipFile(zip_path, 'r') as zipf:
                zipf.extractall()
            extracted = next(Path().glob('vosk-model-*'))
            extracted.rename(target_dir)
            zip_path.unlink()
            print(f"✔ {cfg['name']} model ready at {target_dir}/\n")

    def record_audio(self) -> list:
        self._wait_for_space("▶ Press [space] to START recording…")
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.AUDIO_RATE,
            input=True,
            frames_per_buffer=self.CHUNK_SIZE
        )
        print("● Recording… Press [space] to STOP.")
        frames = []
        stop_flag = threading.Event()
        def stopper():
            self._wait_for_space("")
            stop_flag.set()
        threading.Thread(target=stopper, daemon=True).start()
        while not stop_flag.is_set():
            frames.append(stream.read(self.CHUNK_SIZE, exception_on_overflow=False))
        stream.stop_stream()
        stream.close()
        pa.terminate()
        print("■ Recording stopped.\n")
        return frames

    def _debug_scores(self, results: dict):
        art = r"""
       (\_/)
      ( •_•)  “Debug Mode Activated!”
     <)   )\   Here’s what each model heard:
      /   \ 
        """
        print(art)
        for key, data in results.items():
            name = self.lang_config[key]['name']
            text = data['text'] or '[no words]'
            count = data['count']
            print(f"   • {name:<7} : {count:>2} words  → “{text}”")
        print("\n(＾◡＾)╯ Language scoring complete!\n")

    def detect_language(self, frames: list, debug: bool = None) -> str:
        """
        Identify the language based on processed frames.
        If `debug` is provided, it overrides the engine's debug setting.
        """
        results = {}
        for key, cfg in self.lang_config.items():
            model = Model(str(cfg['dir']))
            rec = KaldiRecognizer(model, self.AUDIO_RATE)
            with tqdm(frames, desc=f"{cfg['name']} processing (＾ω＾)", unit="chunk", ncols=60, ascii=True) as bar:
                for frame in bar:
                    rec.AcceptWaveform(frame)
            text = json.loads(rec.FinalResult()).get('text', '')
            results[key] = {'text': text, 'count': len(text.split())}
        # Determine debug flag
        use_debug = self.debug if debug is None else debug
        if use_debug:
            self._debug_scores(results)
        return max(results, key=lambda k: results[k]['count'])

    def transcribe(self, frames: list, lang_key: str) -> str:
        cfg = self.lang_config[lang_key]
        model = Model(str(cfg['dir']))
        rec = KaldiRecognizer(model, self.AUDIO_RATE)
        with tqdm(frames, desc=f"Transcribing {cfg['name']} (≧◡≦)", unit="chunk", ncols=60, ascii=True) as bar:
            for frame in bar:
                rec.AcceptWaveform(frame)
        return json.loads(rec.FinalResult()).get('text', '')

    def run(self):
        # Show waifu art
        self.show_waifu()
        # Download models if needed
        self.ensure_models()
        # Record audio
        frames = self.record_audio()
        # Detect language
        lang = self.detect_language(frames)
        print(f"🎯 Detected language: {self.lang_config[lang]['name']}\n")
        # Transcribe
        transcript = self.transcribe(frames, lang)
        print("📝 Final Transcription:", transcript)
