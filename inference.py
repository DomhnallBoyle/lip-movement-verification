import argparse
from pathlib import Path

import numpy as np

import encoder.inference as encoder
import synthesizer.inference as _synthesiser
from encoder_preprocess_2 import get_cfe_features


def main(args):
    encoder.load_model(Path(args.encoder_model_path))
    synthesiser = _synthesiser.Synthesizer(Path(args.synthesiser_model_path))
    synthesiser.load()

    videos_directory = Path(args.videos_directory)
    video_paths = videos_directory.glob('*.mp4')
    phrases_path = videos_directory.joinpath('phrases.txt')
    with phrases_path.open('r') as f:
        phrases = f.read().splitlines()

    for i, video_path in enumerate(video_paths):
        # get speaker embeddings first
        cfe_features = get_cfe_features(video_path, args.cfe_host, args.cfe_verify, debug=True)
        cfe_features = np.expand_dims(cfe_features, axis=0)  # add extra dimension to represent batch
        speaker_embeddings = encoder.embed_frames_batch(cfe_features)[0]
        speaker_embeddings = [speaker_embeddings] * len(phrases)

        # get new synethesised features
        new_cfe_features = synthesiser.synthesize_spectrograms(phrases, speaker_embeddings)
        assert len(new_cfe_features) == len(phrases)
        for j, cfe_features in enumerate(new_cfe_features):
            cfe_features = cfe_features.T
            cfe_features_save_path = videos_directory.joinpath(f'S{i+1}_P{j+1}.npy')
            np.save(cfe_features_save_path, cfe_features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('encoder_model_path')
    parser.add_argument('synthesiser_model_path')
    parser.add_argument('videos_directory')
    parser.add_argument('--cfe_host', default='http://0.0.0.0:5001')
    parser.add_argument('--cfe_verify', default=None)

    main(parser.parse_args())
