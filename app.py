from flask import Flask, request, send_file
import librosa
import soundfile as sf
import numpy as np
import os
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/process-audio', methods=['POST'])
def process_audio():
    try:
        # Get parameters from request
        valence = float(request.form.get('valence', 0.5))  # Default to 0.5 if not provided
        arousal = float(request.form.get('arousal', 0.5))  # Default to 0.5 if not provided
        
        # Get the audio file
        if 'audio' not in request.files:
            return {'error': 'No audio file provided'}, 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return {'error': 'No selected file'}, 400
            
        if not audio_file.filename.endswith('.wav'):
            return {'error': 'File must be a WAV file'}, 400

        # Save the uploaded file temporarily
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio_file.save(temp_input.name)
        
        # Load the audio file
        y, sr = librosa.load(temp_input.name)
        
        # Calculate pitch shift based on valence (0-1)
        # Lower valence = lower pitch
        pitch_shift = (valence - 0.5) * 12  # Scale to ±6 semitones
        
        # Calculate speed factor based on arousal (0-1)
        # Lower arousal = slower speed
        speed_factor = 0.5 + (arousal * 0.5)  # Range from 0.5x to 1x speed
        
        # Apply pitch shift
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
        
        # Apply speed change
        y_modified = librosa.effects.time_stretch(y_shifted, rate=speed_factor)
        
        # Save the processed audio
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_output.name, y_modified, sr)
        
        # Clean up the temporary input file
        os.unlink(temp_input.name)
        
        # Send the processed file
        return send_file(
            temp_output.name,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='processed_audio.wav'
        )
        
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True) 