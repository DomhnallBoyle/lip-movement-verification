"""Preprocess videos to CFE arks"""
import argparse
import io
import multiprocessing
import os
import tempfile
import subprocess
import sys
import time
import urllib3
from pathlib import Path

import matplotlib
import cv2
import kaldi_io
import matplotlib.pyplot as plt
import numpy as np
import requests
from tqdm import tqdm

sys.path.append('/home/domhnall/Repos/visual-dtw/app')
from main.utils.pre_process import add_deltas_to_signal

# matplotlib.use('TkAgg')
urllib3.disable_warnings(urllib3.exceptions.SecurityWarning)

DELTA_WINDOW_SIZE = 100
NUM_INITIAL_FEATURES = 283
NUM_DELTAS_FEATURES = NUM_INITIAL_FEATURES * 2
NUM_DELTA_DELTAS_FEATURES = NUM_DELTAS_FEATURES * 2


def read_matrix_ark(f):
    for key, matrix in kaldi_io.read_mat_ark(f):
        return matrix


def get_cfe_features(video_path, host, verify=None, compute_deltas=False, debug=False):
    retry_count = 3

    while retry_count > 0:
        with open(video_path, 'rb') as f:
            try:
                response = requests.post(
                    f'{host}/api/v1/extract/',
                    files={'video': io.BytesIO(f.read())},
                    verify=verify
                )
            except requests.exceptions.ConnectionError as e:
                time.sleep(1)
                retry_count -= 1
                if debug:
                    print(e)
                continue

            # CFE should return file
            if response.headers['Content-Type'] == 'application/json':
                return None

        # temporary file deletes itself after context manager closes
        with tempfile.TemporaryFile() as f:
            f.write(response.content)

            # point to beginning of file after write
            f.seek(0)

            features = read_matrix_ark(f)
            if debug:
                print('Shape:', features.shape)

            if compute_deltas:
                features = add_deltas_to_signal(features, DELTA_WINDOW_SIZE, 0)
                if debug:
                    print('Shape w/ deltas:', features.shape)

            return features.astype(np.float32)

    return None


def preprocess_speakers(process_index: int,
                        speaker_dirs: list,
                        ark_path: Path,
                        cfe_host: str,
                        cfe_verify: str,
                        compute_deltas: bool,
                        debug: bool):
    for speaker_dir in tqdm(speaker_dirs):
        speaker_name = speaker_dir.name
        speaker_ark_dir = ark_path.joinpath(speaker_name)
        speaker_ark_dir.mkdir(exist_ok=True)

        # get all the video clips for a speaker
        video_clip_paths = []
        for video_path in speaker_dir.glob('*'):
            video_clip_paths.extend(video_path.glob('*.mp4'))

        already_processed_path = speaker_ark_dir.joinpath('processed.txt')
        if already_processed_path.exists():
            with already_processed_path.open('r') as f:
                already_processed = f.read().splitlines()
        else:
            already_processed = []

        for video_clip_path in video_clip_paths:
            if str(video_clip_path) in already_processed:
                continue

            features = get_cfe_features(video_clip_path, cfe_host, cfe_verify, compute_deltas, debug)
            if features is not None:
                video_id, video_clip_id = video_clip_path.parts[-2:]
                # numpy_clip_id = video_clip_id.replace('.mp4', '.npy')
                numpy_clip_id = video_clip_id.replace('.mp4', '.npz')
                features_save_path = speaker_ark_dir.joinpath(f'{video_id}_{numpy_clip_id}')

                # np.save(features_save_path, features)

                # .npz is smaller for deltas features
                np.savez_compressed(features_save_path, ark=features)
            else:
                features_save_path = None

            with already_processed_path.open('a') as f:
                f.write(f'{video_clip_path}\n')

            if debug:
                print(process_index, video_clip_path, features_save_path)


def divide_speakers(speaker_dirs, num_processes):
    # divide speaker dirs between processes
    process_speakers = []
    num_speakers = len(speaker_dirs)
    subset_size = num_speakers // num_processes
    for i in range(num_processes):
        speaker_start = i * subset_size
        if i == num_processes - 1:
            speakers = speaker_dirs[speaker_start:]
        else:
            speakers = speaker_dirs[speaker_start:speaker_start + subset_size]

        process_speakers.append([i + 1, speakers])
    assert sum([len(p[1]) for p in process_speakers]) == num_speakers

    return process_speakers


def preprocess(args):
    dataset_root = Path(args.dataset_root)
    speaker_dirs = list(dataset_root.joinpath('dev', 'mp4').glob('*'))

    ark_path = Path(args.arks_root)
    ark_path.mkdir(exist_ok=True)

    process_speakers = divide_speakers(speaker_dirs, args.num_processes)
    for p in process_speakers:
        p.extend([ark_path, args.cfe_host, args.cfe_verify, args.compute_deltas, args.debug])  # add some extra data

    with multiprocessing.Pool(processes=args.num_processes) as pool:
        pool.starmap(preprocess_speakers, process_speakers)


def analyse_video_paths(process_index: int, speaker_dirs: list):
    durations = []
    durations_append = durations.append
    for speaker_dir in speaker_dirs:
        for video_path in speaker_dir.glob('**/*.mp4'):
            # extract video duration
            result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                     "format=duration", "-of",
                                     "default=noprint_wrappers=1:nokey=1", str(video_path)],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            duration = float(result.stdout)
            durations_append(duration)

    return durations


def analyse(args):
    video_durations_path = Path('video_durations.npy')

    def show_hist(_durations):
        plt.hist(_durations, bins=np.arange(_durations.min(), _durations.max()+1))
        plt.show()

    if video_durations_path.exists():
        durations = np.load(str(video_durations_path))
    else:
        dataset_root = Path(args.dataset_root)
        speaker_dirs = list(dataset_root.joinpath('dev', 'mp4').glob('*'))

        process_speakers = divide_speakers(speaker_dirs, args.num_processes)
        durations = []
        with multiprocessing.Pool(processes=args.num_processes) as pool:
            results = pool.starmap(analyse_video_paths, process_speakers)
            for ds in results:
                durations.extend(ds)

        durations = np.asarray(durations)
        np.save(str(video_durations_path), durations)

    show_hist(durations)


def _compute_deltas(process_index: int, speaker_dirs: list):
    # npz requires less storage than < npy

    for speaker_dir in tqdm(speaker_dirs):
        if not speaker_dir.is_dir():
            continue

        ark_paths = []
        for extension in ['*.npy', '*.npz']:
            ark_paths.extend(list(speaker_dir.glob(extension)))

        for ark_path in ark_paths:
            m = np.load(str(ark_path))
            is_npz = ark_path.name.endswith('.npz')
            if is_npz:
                m = m['ark']

            num_features = m.shape[1]
            if num_features == NUM_INITIAL_FEATURES:
                new_m = add_deltas_to_signal(m, DELTA_WINDOW_SIZE, 0)  # compute delta deltas
            elif num_features == NUM_DELTA_DELTAS_FEATURES:
                new_m = m[:, :NUM_DELTAS_FEATURES]

            assert new_m.shape[1] == NUM_DELTAS_FEATURES
            assert np.array_equal(m[:, :NUM_INITIAL_FEATURES], new_m[:, :NUM_INITIAL_FEATURES])

            # save new ark in compressed format
            save_path = str(ark_path)
            if not is_npz:
                os.remove(save_path)  # remove old ark npy file
                save_path = save_path.replace('.npy', '.npz')

            np.savez_compressed(save_path, ark=new_m)


def compute_deltas(args):
    arks_root = Path(args.arks_root)
    speaker_dirs = list(arks_root.glob('*'))

    process_speakers = divide_speakers(speaker_dirs, args.num_processes)

    with multiprocessing.Pool(processes=args.num_processes) as pool:
        pool.starmap(_compute_deltas, process_speakers)


def main(args):
    f = {
        'preprocess': preprocess,
        'analyse': analyse,
        'compute_deltas': compute_deltas
    }
    f[args.run_type](args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='run_type')

    parser_1 = sub_parsers.add_parser('preprocess')
    parser_1.add_argument('dataset_root')
    parser_1.add_argument('arks_root')
    parser_1.add_argument('--num_processes', type=int, default=4)
    parser_1.add_argument('--cfe_host', default='http://0.0.0.0:5001')
    parser_1.add_argument('--cfe_verify', default=None)
    parser_1.add_argument('--compute_deltas', action='store_true')
    parser_1.add_argument('--debug', action='store_true')

    parser_2 = sub_parsers.add_parser('analyse')
    parser_2.add_argument('dataset_root')
    parser_2.add_argument('--num_processes', type=int, default=4)

    parser_3 = sub_parsers.add_parser('compute_deltas')
    parser_3.add_argument('arks_root')
    parser_3.add_argument('--num_processes', type=int, default=4)

    main(parser.parse_args())
