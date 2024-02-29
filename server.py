import argparse
import json
import soundfile as sf
import tempfile
from pathlib import Path

import numpy as np
from flask import Flask, request, send_file

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

app = Flask(__name__)
synthesiser = None


@app.route('/tts', methods=['POST'])
def tts():
    audio_file = request.files['audio']
    text = request.form['text']

    with tempfile.NamedTemporaryFile(suffix='.wav') as f1:
        with open(f1.name, 'wb') as f2:
            f2.write(audio_file.read())
        f1.seek(0)

        # get speaker embedding
        preprocessed_wav = encoder.preprocess_wav(f1.name)
        speaker_embedding = encoder.embed_utterance(preprocessed_wav)

    # synthesise from text
    specs = synthesiser.synthesize_spectrograms([text], [speaker_embedding])
    spec = specs[0]

    # generate audio from spectrogram
    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = np.pad(generated_wav, (0, synthesiser.sample_rate), mode='constant')
    generated_wav = encoder.preprocess_wav(generated_wav)

    with tempfile.NamedTemporaryFile(suffix='.wav') as f:
        sf.write(f.name, generated_wav.astype(np.float32), synthesiser.sample_rate)

        return send_file(f.name, as_attachment=True)


def main(args):
    global synthesiser

    encoder.load_model(Path(args.encoder_model_path), device='cpu')
    synthesiser = Synthesizer(Path(args.synthesiser_model_path), device='cpu')
    vocoder.load_model(Path(args.vocoder_model_path), device='cpu')

    app.run(host='0.0.0.0', port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--encoder_model_path', default='encoder/saved_models/pretrained.pt')
    parser.add_argument('--synthesiser_model_path', default='synthesizer/saved_models/pretrained.pt')
    parser.add_argument('--vocoder_model_path', default='vocoder/saved_models/vocoder_1159k.pt')

    main(parser.parse_args())
