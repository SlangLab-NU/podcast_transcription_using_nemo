from torch.utils.data import DataLoader
import math
import torch

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

    @torch.no_grad()
    def transcribe_buffers(self, buffers, merge=True, buffer_offset=0):
        self.buffers = buffers
        self.data_layer.set_signal(buffers[:])
        self._get_batch_preds()
        return self.decode_final(merge, buffer_offset)

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

    def decode_final(self, merge=True, buffer_offset=0):
        self.unmerged = []
        self.toks_unmerged = []
        self.time_stamps = []
        # index for the first token corresponding to a chunk of audio would be len(decoded) - 1 - delay
        delay = math.ceil((self.chunk_len + (self.buffer_len -
                          self.chunk_len) / 2) / self.model_stride_in_sec)

        decoded_frames = []
        all_toks = []
        all_times = []
        for pred in self.all_preds:
            ids, toks, times = self._greedy_decoder(
                pred, self.asr_model.tokenizer)
            decoded_frames.append(ids)
            all_toks.append(toks)
            all_times.append(times)

        for i, decoded in enumerate(decoded_frames):
            start_index = len(decoded) - 1 - delay
            end_index = start_index + self.n_tokens_per_chunk
            self.unmerged += decoded[start_index:end_index]
            adjusted_times = [t * self.model_stride_in_sec +
                              buffer_offset for t in all_times[i][start_index:end_index]]
            self.time_stamps += adjusted_times

            # print(
            #     f"Buffer {i}, Buffer offset: {buffer_offset}, Adjusted times: {adjusted_times}")

        if not merge:
            text, time_stamps = self.combine_tokens_into_words(
                self.unmerged, self.time_stamps)
            return text, time_stamps

        merged_text = self.greedy_merge(self.unmerged)
        return merged_text, self.get_time_stamps(self.time_stamps)

    def _greedy_decoder(self, preds, tokenizer):
        s = []
        ids = []
        times = []
        for i in range(preds.shape[0]):
            if preds[i] == self.blank_id:
                s.append("_")
            else:
                pred = preds[i]
                s.append(tokenizer.ids_to_tokens([pred.item()])[0])
            ids.append(preds[i].item())
            times.append(i)
        return ids, s, times

    def greedy_merge(self, preds):
        decoded_prediction = []
        previous = self.blank_id
        for p in preds:
            if (p != previous or previous == self.blank_id) and p != self.blank_id:
                decoded_prediction.append(int(p))
            previous = p
        hypothesis = self.asr_model.tokenizer.ids_to_text(decoded_prediction)
        return hypothesis

    def combine_tokens_into_words(self, ids, time_stamps):
        words = []
        start_times = []
        end_times = []
        current_word = []
        current_start_time = None

        tokens = self.convert_ids_to_text(ids)

        for i, token in enumerate(tokens):
            if token == "<UNK>":
                continue  # Skip unknown tokens

            if token.startswith("▁"):  # Using the special character for word boundary
                if current_word:
                    words.append("".join(current_word))
                    start_times.append(current_start_time)
                    end_times.append(time_stamps[i - 1])
                current_word = [token[1:]]  # Remove leading special character
                current_start_time = time_stamps[i]
            else:
                if current_start_time is None:
                    current_start_time = time_stamps[i]
                current_word.append(token)

        if current_word:  # Append the last word if exists
            words.append("".join(current_word))
            start_times.append(current_start_time)
            end_times.append(time_stamps[-1])

        return words, list(zip(start_times, end_times))

    def convert_ids_to_text(self, ids):
        tokens = []
        for id_ in ids:
            try:
                tokens.append(
                    self.asr_model.tokenizer.ids_to_tokens([int(id_)])[0])
            except KeyError:
                # Handle special or unknown tokens
                tokens.append('<UNK>')
        return tokens

    def get_time_stamps(self, time_stamps):
        start_times = time_stamps
        end_times = [t + self.model_stride_in_sec for t in start_times]
        return list(zip(start_times, end_times))


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
