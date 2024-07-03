# A simple iterator class to return successive chunks of samples
import numpy as np


class AudioChunkIterator():
    """
    A simple iterator class to return successive chunks of samples.
    """

    def __init__(self, samples, sample_rate, chunk_len_in_sec):
        self._samples = samples
        self._chunk_len = chunk_len_in_sec * sample_rate
        self._start = 0
        self.output = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.output:
            raise StopIteration
        last = int(self._start + self._chunk_len)
        if last <= len(self._samples):
            chunk = self._samples[self._start: last]
            self._start = last
        else:
            chunk = np.zeros([int(self._chunk_len)], dtype='float32')
            samp_len = len(self._samples) - self._start
            chunk[0:samp_len] = self._samples[self._start:len(self._samples)]
            self.output = False

        return chunk
