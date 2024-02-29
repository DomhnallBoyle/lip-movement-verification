import argparse
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from encoder import inference as encoder
from encoder_preprocess_2 import get_cfe_features


class Dataset:

    def __init__(self, root_dir: Path, preprocessed_dir: str):
        self.root_dir = root_dir
        self.preprocessed_dir = self.root_dir.joinpath(preprocessed_dir); self.preprocessed_dir.mkdir(exist_ok=True)

        self.metadata = self.preprocessed_dir.joinpath('train.csv')
        self.arks_dir = self.preprocessed_dir.joinpath('arks'); self.arks_dir.mkdir(exist_ok=True)
        self.embeds_dir = self.preprocessed_dir.joinpath('embeds'); self.embeds_dir.mkdir(exist_ok=True)


class SRAVI(Dataset):

    def __init__(self, root_path: str):
        super().__init__(Path(root_path), 'cloning_synthesiser_dataset')
        self.videos_path = self.root_dir.joinpath('videos')
        self.phrase_regex = r'SRAVIExtended.+P(\d+)-.+'
        self.phrases = ["What's the plan?", 'I feel depressed', 'Call my family', "I'm hot", "I'm cold",
                        'I feel anxious', 'What time is it?', "I don't want that", 'How am I doing?',
                        'I need the bathroom', "I'm comfortable", "I'm thirsty", "It's too bright", "I'm in pain",
                        'Move me', "It's too noisy", 'Doctor', "I'm hungry", 'Can I have a cough?', 'I am scared',
                        'My head hurts', 'My arm is sore', 'My leg is sore', 'My chest feels tight', 'My throat is dry',
                        'I have toothache', 'My ear is sore', 'My skin is itchy', 'My nose is runny', "I'm tired",
                        'How much is that?', 'Can I have a bag?', 'Can you help me?', 'I would like a coffee',
                        'I would like tea', 'How are you?', 'I am feeling good', 'I feel great!',
                        "It's good to see you", 'Thank you', "Let's go out", 'I need fresh air',
                        'Could you open the door?', 'I agree', 'I disagree', 'Please talk slower',
                        'Can you speak louder?', 'Can you speaker more quietly?', 'Can you put the light on?',
                        "I'm busy", "I'm available", "It's my birthday", 'That sounds good', "I'm happy", "I'm sad",
                        "I don't want to do that", 'I want to be alone', 'I need quiet', 'I need a rest', 'This week',
                        'Next week', 'This month', 'Next month', 'This year', 'Next year', 'This morning',
                        'This afternoon', 'This evening', "It's important", "It's urgent", 'Can you call', 'My mother',
                        'My father', 'My brother', 'My sister', 'My son', 'My daughter', 'My relative',
                        'Better than yesterday', 'Worse than yesterday', "I'm fed up", 'This is boring',
                        'When will it end?', 'When is the next one?', 'I want a shower', 'I need headphones',
                        'Can you turn on the tv?', 'Can you pass my book?', 'Can we go outside?',
                        'Can I have a blanket?', 'How are you today?', 'Tell me about your day?', 'I want a haircut',
                        'Is it warm outside?', 'Is it cold outside?', 'Have you plans for the weekend?',
                        "I'm going on holiday", 'I feel lonely', 'This is the way', 'SRAVI is great!']

    def get_speaker_paths(self):
        return self.videos_path.glob('*')

    def get_phrase(self, video_path):
        basename = video_path.name.replace('.mp4', '')
        phrase_id = int(re.match(self.phrase_regex, basename).groups()[0])

        return self.phrases[phrase_id-1]

    def extract_features(self, cfe_host: str, cfe_verify=None, **kwargs):
        """Grab the CFE features for each video first. Grab embeddings after"""
        for speaker_path in self.get_speaker_paths():
            print(f'\n{speaker_path}')
            for speaker_video_path in tqdm(speaker_path.glob('*.mp4')):
                phrase = self.get_phrase(speaker_video_path)

                features = get_cfe_features(speaker_video_path, cfe_host, cfe_verify, debug=True)
                if features is None:
                    continue

                speaker_video_basename = speaker_video_path.name
                features_save_path = self.arks_dir.joinpath(speaker_video_basename.replace('mp4', 'npy'))
                embeddings_save_path = self.embeds_dir.joinpath(speaker_video_basename.replace('mp4', 'npy'))

                # save cfe features
                np.save(str(features_save_path), features)
                num_feature_frames = features.shape[0]

                # append to metadata
                with self.metadata.open('a') as f:
                    f.write(f'{speaker_video_path},{features_save_path},{embeddings_save_path},{0},{num_feature_frames},{phrase}\n')

    def extract_embeddings(self, encoder_model_path, **kwargs):
        encoder.load_model(Path(encoder_model_path))

        with self.metadata.open('r') as f:
            for line in tqdm(f.read().splitlines()):
                features_path, embeddings_path = line.split(',')[1:3]
                features = np.load(features_path)
                features = np.expand_dims(features, axis=0)  # add extra dimension to represent batch
                embeddings = encoder.embed_frames_batch(features)[0]

                np.save(embeddings_path, embeddings)


def main(args):
    dataset = SRAVI(args.dataset_root)

    if args.run_type == 'extract_features':
        dataset.extract_features(**args.__dict__)
    elif args.run_type == 'extract_embeddings':
        dataset.extract_embeddings(**args.__dict__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root')

    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('extract_features')
    parser_1.add_argument('--cfe_host', default='http://0.0.0.0:5000')
    parser_1.add_argument('--cfe_verify', default=None)

    parser_2 = sub_parsers.add_parser('extract_embeddings')
    parser_2.add_argument('encoder_model_path')

    main(parser.parse_args())
