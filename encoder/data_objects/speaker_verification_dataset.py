import os
from encoder.data_objects.random_cycler import RandomCycler
from encoder.data_objects.speaker_batch import SpeakerBatch
from encoder.data_objects.speaker import Speaker, Speaker2
from encoder.params_data import partials_n_frames
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

import numpy as np

# TODO: improve with a pool of speakers for data efficiency


class SpeakerVerificationDataset(Dataset):
    def __init__(self, datasets_root: Path):
        self.root = datasets_root
        speaker_dirs = [f for f in self.root.glob("*") if f.is_dir()]
        if len(speaker_dirs) == 0:
            raise Exception("No speakers found. Make sure you are pointing to the directory "
                            "containing all preprocessed speaker directories.")
        # self.speakers = [Speaker2(speaker_dir) for speaker_dir in speaker_dirs]
        # self.speaker_cycler = RandomCycler(self.speakers)

        # remove any speakers with no samples or invalid samples i.e. < n_frames
        all_excluded_utterances = {speaker_dir: [] for speaker_dir in speaker_dirs}
        for speaker_dir in speaker_dirs:
            samples = list(speaker_dir.glob('*.npy'))
            if not samples:
                print(speaker_dir.name, 'has no samples')
                del all_excluded_utterances[speaker_dir]
                continue

            for sample in samples:
                sample_size = sample.stat().st_size
                if sample_size < 180000:  # bytes ~= 160 frames i.e. video too short
                    d = np.load(sample)
                    print(speaker_dir.name, sample.name, f'only has {d.shape[0]} frames')
                    all_excluded_utterances[speaker_dir].append(sample.name)

        self.speakers = [Speaker2(speaker_dir, excluded_utterances)
                         for speaker_dir, excluded_utterances in all_excluded_utterances.items()]
        self.speaker_cycler = RandomCycler(self.speakers)

    def __len__(self):
        return int(1e10)
        
    def __getitem__(self, index):
        return next(self.speaker_cycler)
    
    def get_logs(self):
        log_string = ""
        for log_fpath in self.root.glob("*.txt"):
            with log_fpath.open("r") as log_file:
                log_string += "".join(log_file.readlines())
        return log_string
    
    
class SpeakerVerificationDataLoader(DataLoader):
    def __init__(self, dataset, speakers_per_batch, utterances_per_speaker, sampler=None, 
                 batch_sampler=None, num_workers=0, pin_memory=False, timeout=0, 
                 worker_init_fn=None):
        self.utterances_per_speaker = utterances_per_speaker

        super().__init__(
            dataset=dataset, 
            batch_size=speakers_per_batch, 
            shuffle=False, 
            sampler=sampler, 
            batch_sampler=batch_sampler, 
            num_workers=num_workers,
            collate_fn=self.collate, 
            pin_memory=pin_memory, 
            drop_last=False, 
            timeout=timeout, 
            worker_init_fn=worker_init_fn
        )

    def collate(self, speakers):
        return SpeakerBatch(speakers, self.utterances_per_speaker, partials_n_frames) 
