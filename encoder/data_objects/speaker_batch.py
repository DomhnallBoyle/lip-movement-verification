import multiprocessing

import numpy as np
from typing import List
from encoder.data_objects.speaker import Speaker


def load_speaker_partials(process_index: int,
                          speakers: List[Speaker],
                          utterances_per_speaker: int,
                          n_frames: int):
    return {
        s.name: s.random_partial(utterances_per_speaker, n_frames)
        for s in speakers
    }


class SpeakerBatch:

    def __init__(self, speakers: List[Speaker], utterances_per_speaker: int, n_frames: int):
        self.speakers = speakers
        self.num_speakers = len(speakers)
        self.num_processes = 4
        self.utterances_per_speaker = utterances_per_speaker
        self.n_frames = n_frames
        self.speaker_names = [s.name for s in speakers]

        self.partials = {s: s.random_partial(utterances_per_speaker, n_frames) for s in speakers}
        # self.partials = self._get_partials()

        # Array of shape (n_speakers * n_utterances, n_frames, mel_n), e.g. for 3 speakers with
        # 4 utterances each of 160 frames of 40 mel coefficients: (12, 160, 40)
        self.data = np.array([frames for s in speakers for _, frames, _ in self.partials[s]])  # takes ~ 0.2 seconds
        # self.data = np.array([frames for name in self.speaker_names for _, frames, _ in self.partials[name]])

    def _get_partials(self):
        # divide speakers between processes
        process_speakers = []
        subset_size = self.num_speakers // self.num_processes
        for i in range(self.num_processes):
            speaker_start = i * subset_size
            if i == self.num_processes - 1:
                speakers = self.speakers[speaker_start:]
            else:
                speakers = self.speakers[speaker_start:speaker_start + subset_size]
            process_speakers.append([i+1, speakers, self.utterances_per_speaker, self.n_frames])
        assert sum([len(p[1]) for p in process_speakers]) == self.num_speakers

        speakers_d = {}
        with multiprocessing.pool.ThreadPool(processes=self.num_processes) as pool:
            results = pool.starmap(load_speaker_partials, process_speakers)
            for partials in results:
                speakers_d.update(partials)

        assert self.num_speakers == len(list(speakers_d.keys()))

        return speakers_d
