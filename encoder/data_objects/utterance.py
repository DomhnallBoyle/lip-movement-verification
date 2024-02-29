import numpy as np


class Utterance:
    def __init__(self, frames_fpath, wave_fpath):
        self.frames_fpath = frames_fpath
        self.wave_fpath = wave_fpath
        
    def get_frames(self):
        return np.load(self.frames_fpath)

    def random_partial(self, n_frames):
        """
        Crops the frames into a partial utterance of n_frames
        
        :param n_frames: The number of frames of the partial utterance
        :return: the partial utterance frames and a tuple indicating the start and end of the 
        partial utterance in the complete utterance.
        """
        frames = self.get_frames()  # taking ~0.02 seconds per sample = 12.8 seconds (640 samples)

        if frames.shape[0] == n_frames:
            start = 0
        else:
            start = np.random.randint(0, frames.shape[0] - n_frames)
        end = start + n_frames

        return frames[start:end], (start, end)


class Utterance2(Utterance):
    __slots__ = ('ark_path',)

    def __init__(self, ark_path):
        super().__init__(None, None)
        self.ark_path = ark_path

    def get_frames(self):
        return np.load(self.ark_path)
