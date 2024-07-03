from torch.utils.data import DataLoader
import math
import torch
import matplotlib.pyplot as plt

from AudioBuffersDataLayer import AudioBuffersDataLayer


class ChunkBufferDecoder:
    """
    A class to decode audio buffers using a sliding window approach.
    """

    def __init__(self, asr_model, stride, chunk_len_in_sec=1, buffer_len_in_sec=3):
        self.asr_model = asr_model
        self.asr_model.eval()
        self.data_layer = AudioBuffersDataLayer()
        self.data_loader = DataLoader(
            self.data_layer, batch_size=1, collate_fn=speech_collate_fn)
        self.buffers = []
        self.all_preds = []
        self.chunk_len = chunk_len_in_sec
        self.buffer_len = buffer_len_in_sec
        assert (chunk_len_in_sec <= buffer_len_in_sec)

        feature_stride = asr_model._cfg.preprocessor['window_stride']
        self.model_stride_in_sec = feature_stride * stride
        self.n_tokens_per_chunk = math.ceil(
            self.chunk_len / self.model_stride_in_sec)
        self.blank_id = len(asr_model.decoder.vocabulary)
        self.plot = False

    @torch.no_grad()
    def transcribe_buffers(self, buffers, merge=True, plot=False):
        self.plot = plot
        self.buffers = buffers
        self.data_layer.set_signal(buffers[:])
        self._get_batch_preds()
        return self.decode_final(merge)

    def _get_batch_preds(self):

        device = self.asr_model.device
        for batch in iter(self.data_loader):

            audio_signal, audio_signal_len = batch

            audio_signal, audio_signal_len = audio_signal.to(
                device), audio_signal_len.to(device)
            log_probs, encoded_len, predictions = self.asr_model(
                input_signal=audio_signal, input_signal_length=audio_signal_len)
            preds = torch.unbind(predictions)
            for pred in preds:
                self.all_preds.append(pred.cpu().numpy())

    def decode_final(self, merge=True, extra=0):
        self.unmerged = []
        self.toks_unmerged = []
        # index for the first token corresponding to a chunk of audio would be len(decoded) - 1 - delay
        delay = math.ceil((self.chunk_len + (self.buffer_len -
                          self.chunk_len) / 2) / self.model_stride_in_sec)

        decoded_frames = []
        all_toks = []
        for pred in self.all_preds:
            ids, toks = self._greedy_decoder(pred, self.asr_model.tokenizer)
            decoded_frames.append(ids)
            all_toks.append(toks)

        for decoded in decoded_frames:
            self.unmerged += decoded[len(decoded) - 1 - delay:len(
                decoded) - 1 - delay + self.n_tokens_per_chunk]
        if self.plot:
            for i, tok in enumerate(all_toks):
                plt.plot(self.buffers[i])
                plt.show()
                print("\nGreedy labels collected from this buffer")
                print(tok[len(tok) - 1 - delay:len(tok) -
                      1 - delay + self.n_tokens_per_chunk])
                self.toks_unmerged += tok[len(tok) - 1 - delay:len(
                    tok) - 1 - delay + self.n_tokens_per_chunk]
            print("\nTokens collected from successive buffers before CTC merge")
            print(self.toks_unmerged)

        if not merge:
            return self.unmerged
        return self.greedy_merge(self.unmerged)

    def _greedy_decoder(self, preds, tokenizer):
        s = []
        ids = []
        for i in range(preds.shape[0]):
            if preds[i] == self.blank_id:
                s.append("_")
            else:
                pred = preds[i]
                s.append(tokenizer.ids_to_tokens([pred.item()])[0])
            ids.append(preds[i])
        return ids, s

    def greedy_merge(self, preds):
        decoded_prediction = []
        previous = self.blank_id
        for p in preds:
            if (p != previous or previous == self.blank_id) and p != self.blank_id:
                decoded_prediction.append(p.item())
            previous = p
        hypothesis = self.asr_model.tokenizer.ids_to_text(decoded_prediction)
        return hypothesis


def speech_collate_fn(batch: list) -> tuple:
    """
    Collate batch of audio signals into a single tensor with their lengths.

    Args:
        batch (FloatTensor, LongTensor):  A tuple of tuples of signal, signal lengths.
        This collate func assumes the signals are 1d torch tensors (i.e. mono audio).

    Returns:
        (FloatTensor, LongTensor): A tuple containing the signals stacked as a tensor and the lengths.
    """

    _, audio_lengths = zip(*batch)

    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()

    audio_signal = []
    for sig, sig_len in batch:
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None

    return audio_signal, audio_lengths
