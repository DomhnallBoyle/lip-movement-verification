import pickle
from pathlib import Path

import lmdb
import numpy as np


class Ark:
    __slots__ = ('channels', 'size', 'image')

    def __init__(self, a):
        self.channels = 2
        self.size = a.shape[:2]
        self.image = a.tobytes()

    def get_image(self):
        image = np.frombuffer(self.image, dtype=np.uint8)

        return image.reshape(*self.size, self.channels)


def store_many_lmdb(root, lmdb_dir):
    num_samples = len(list(root.glob('**/*.npy')))
    map_size = num_samples * 200000 * 10

    env = lmdb.open(f'{lmdb_dir}/{num_samples}_lmdb', map_size=map_size)

    with env.begin(write=True) as txn:
        for ark in root.glob('**/*.npy'):
            value = Ark(np.load(ark))
            key = str(ark)
            txn.put(key.encode('ascii'), pickle.dumps(value))
    env.close()


if __name__ == '__main__':
    ark_path = Path('tmp')
    ark_path.mkdir(exist_ok=True)

    store_many_lmdb(Path('/media/alex/Storage/Domhnall/datasets/vox_celeb/2/videos/dev/arks'), 'tmp')
