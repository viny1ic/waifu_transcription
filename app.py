from flask import Flask, request, jsonify
from pathlib import Path
import tempfile
import wave

# Import your TranscriptionEngine class
from transcription_engine import TranscriptionEngine

# Base dir where index.html and related files live
BASE_DIR = Path(__file__).parent

# Initialize Flask app with static folder set to serve frontend files
app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path='')

# Instantiate TranscriptionEngine (no waifu art, no debug, small models by default)
engine = TranscriptionEngine(
    base_dir=BASE_DIR,
    waifu=False,
    debug=False,
    big=False
)

@app.route('/')
def index():
    """
    Serve the main frontend page.
    """
    return app.send_static_file('index.html')

@app.route('/api/transcribe', methods=['POST'])
def transcribe_api():
    """
    Receive audio blob in WAV format, run transcription, and return JSON with transcription.
    """
    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({'error': 'No audio file provided'}), 400

    # Save uploaded WAV to a temporary file
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    audio_file.save(temp_wav.name)
    temp_wav.flush()

    # Read WAV frames in chunks
    frames = []
    with wave.open(temp_wav.name, 'rb') as wf:
        while True:
            data = wf.readframes(engine.CHUNK_SIZE)
            if not data:
                break
            frames.append(data)

    # Detect language & transcribe
    lang = engine.detect_language(frames, debug=engine.debug)
    transcript = engine.transcribe(frames, lang)

    # Clean up temp file
    temp_wav.close()

    return jsonify({'transcription': transcript})

if __name__ == '__main__':
    # Run on http://localhost:5000/
    app.run(host='0.0.0.0', port=5000, debug=True)
