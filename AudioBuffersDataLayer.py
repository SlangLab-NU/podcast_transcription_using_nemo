import torch
from nemo.core.classes import IterableDataset


class AudioBuffersDataLayer(IterableDataset):
    """
    A simple iterable dataset class to return a single buffer of samples.
    """

    def __init__(self):
        super().__init__()

    def __iter__(self):
        return self

    def __next__(self):
        if self._buf_count == len(self.signal):
            raise StopIteration
        self._buf_count += 1
        return torch.as_tensor(self.signal[self._buf_count-1], dtype=torch.float32), \
            torch.as_tensor(self.signal_shape[0], dtype=torch.float32)

    def set_signal(self, signals):
        self.signal = signals
        self.signal_shape = self.signal[0].shape
        self._buf_count = 0

    def __len__(self):
        return 1
