from torch.utils.data import DataLoader
import math
import torch
from typing import List, Tuple
import logging

from AudioBuffersDataLayer import AudioBuffersDataLayer


class ChunkBufferDecoder:
    """
    A class to decode audio buffers using a sliding window approach.
    
    Attributes:
        asr_model (torch.nn.Module): The ASR model to use for decoding.
        stride (int): The stride in number of frames between each chunk.
        chunk_len (float): The length of each chunk in seconds.
        buffer_len (float): The length of the buffer in seconds.
        model_stride_in_sec (float): The stride of the model in seconds.
        n_tokens_per_chunk (int): The number of tokens in each chunk.
        blank_id (int): The id of the blank token in the vocabulary.
        data_layer (AudioBuffersDataLayer): The data layer to use for loading audio buffers.
        data_loader (DataLoader): The data loader to use for loading audio buffers.
        buffers (list): The list of audio buffers to decode.
        all_preds (list): The list of all predictions made by the model.
        unmerged (list): The list of unmerged predictions.
        time_stamps (list): The list of time stamps for each prediction.
    """

    def __init__(
            self,
            asr_model: torch.nn.Module,
            stride: float = 1.0,
            chunk_len_in_sec: float = 1.0,
            buffer_len_in_sec: float = 1.0,
            context_len_in_sec: float = 1.0,
    ):
        """
        Initialize the ChunkBufferDecoder.
        
        Args:
            asr_model (torch.nn.Module): The ASR model to use for decoding.
            stride (int): The stride in number of frames between each chunk.
            chunk_len_in_sec (float): The length of each chunk in seconds.
            buffer_len_in_sec (float): The length of the buffer in seconds.
            context_len_in_sec (float): The length of the context in seconds.
        """
        self.asr_model = asr_model
        self.asr_model.eval()
        self.data_layer = AudioBuffersDataLayer()
        self.data_loader = DataLoader(
            self.data_layer, batch_size=1, collate_fn=speech_collate_fn)
        self.buffers = []
        self.all_preds = []
        self.chunk_len = chunk_len_in_sec
        self.buffer_len = buffer_len_in_sec
        self.context_len = context_len_in_sec
        assert (chunk_len_in_sec <= buffer_len_in_sec)

        feature_stride = asr_model._cfg.preprocessor['window_stride']
        self.model_stride_in_sec = feature_stride * stride
        self.n_tokens_per_chunk = math.ceil(
            self.chunk_len / self.model_stride_in_sec)
        self.blank_id = len(asr_model.decoder.vocabulary)
        
    def reset(self):
        """
        Reset the decoder.
        """
        self.buffers = []
        self.all_preds = []

    @torch.no_grad()
    def transcribe_buffers(self, buffers: List[torch.Tensor], merge: bool = True, buffer_offset: float = 0.0) -> Tuple[str, List[Tuple[float, float]]]:
        """
        Transcribe a list of audio buffers.
        
        Args:
            buffers (List[torch.Tensor]): A list of audio buffers to transcribe.
            merge (bool): Whether to merge the predictions.
            buffer_offset (float): The offset in seconds to add to the time stamps.
            
        Returns:
            merged_text (str): The merged text of the transcribed audio buffers.
            time_stamps (List[Tuple[float, float]]): The list of time stamps for each prediction.
        """
        self.buffers = buffers
        self.data_layer.set_signal(buffers[:])
        self._get_batch_preds()
        
        logging.info(f"Processing buffers for transcription, offset: {buffer_offset}s\n")

        merged_text, time_stamps = self.decode_final(merge, buffer_offset)
        
        return merged_text, time_stamps

    def _get_batch_preds(self) -> None:
        """
        Get the predictions for each buffer in the batch.
        """
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

    def decode_final(self, merge: bool = True, buffer_offset: float = 0.0) -> Tuple[str, List[Tuple[float, float]]]:
        """
        Decode the final predictions.
        
        Args:
            merge (bool): Whether to merge the predictions.
            buffer_offset (float): The offset in seconds to add to the time stamps.
            
        Returns:
            merged_text (str): The merged text of the transcribed audio buffers.
            time_stamps (List[Tuple[float, float]]): The list of time stamps for each prediction.
        """
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
            
            # Adjust times to account for the buffer offset
            adjusted_times = [(t * self.model_stride_in_sec) +
                              buffer_offset - self.context_len for t in times]
            
            decoded_frames.append(ids)
            all_toks.append(toks)
            all_times.append(adjusted_times)
            

        for i, decoded in enumerate(decoded_frames):
            start_index = len(decoded) - 1 - delay
            end_index = start_index + self.n_tokens_per_chunk
            self.unmerged += decoded[start_index:end_index]
            self.time_stamps += all_times[i][start_index:end_index]

        if not merge:
            text, time_stamps = self.combine_tokens_into_words(
                self.unmerged, self.time_stamps)

            return text, time_stamps

        merged_text = self.greedy_merge(self.unmerged)
        
        return merged_text, self.get_time_stamps(self.time_stamps)

    def _greedy_decoder(self, preds: torch.Tensor, tokenizer: torch.nn.Module) -> Tuple[List[int], List[str], List[int]]:
        """
        Decode the predictions using a greedy decoder.
        
        Args:
            preds (torch.Tensor): The predictions to decode.
            tokenizer (torch.nn.Module): The tokenizer to use for decoding.
            
        Returns:
            ids (List[int]): The list of ids of the decoded tokens.
            s (List[str]): The list of decoded tokens.
            times (List[int]): The list of time stamps for each token.
        """
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

    def greedy_merge(self, preds: List[int]) -> str:
        """
        Merge the predictions using a greedy approach.
        
        Args:
            preds (List[int]): The list of predictions to merge.
            
        Returns:
            hypothesis (str): The merged hypothesis.
        """
        decoded_prediction = []
        previous = self.blank_id
        for p in preds:
            if (p != previous or previous == self.blank_id) and p != self.blank_id:
                decoded_prediction.append(int(p))
            previous = p
        hypothesis = self.asr_model.tokenizer.ids_to_text(decoded_prediction)

        return hypothesis

    def combine_tokens_into_words(self, ids: List[int], time_stamps: List[int]) -> Tuple[str, List[Tuple[float, float]]]:
        """
        Combine the tokens into words, adjusting for the buffer's offset in the overall audio.
        
        Args:
            ids (List[int]): The list of token ids.
            time_stamps (List[int]): The list of time stamps for each token.
            
        Returns:
            words (str): The words of the transcribed audio.
            time_stamps (List[Tuple[float, float]]): The list of time stamps for each word.
        """
        words = []
        start_times = []
        end_times = []
        current_word = []
        current_start_time = None
        current_end_time = None

        tokens = self.convert_ids_to_text(ids)
        context_offset = self.context_len

        for i, token in enumerate(tokens):
            if token == "<UNK>":
                continue  # Skip unknown tokens

            if token.startswith("▁"):  # Using the special character for word boundary
                if current_word:
                    words.append("".join(current_word))
                    # Adjust start and end times by subtracting the context offset
                    start_times.append(current_start_time - context_offset)
                    # Use current_end_time updated during the processing of the last token of the word
                    end_times.append(current_end_time - context_offset)
                current_word = [token[1:]]  # Remove leading special character
                current_start_time = time_stamps[i]
                current_end_time = time_stamps[i]  # Initialize end time with the start of the new word
            else:
                if current_start_time is None:  # This should only trigger if the first token isn't a word boundary
                    current_start_time = time_stamps[i]
                current_word.append(token)
                current_end_time = time_stamps[i]  # Update end time with each token

        # Append the last word if it exists
        if current_word:
            words.append("".join(current_word))
            start_times.append(current_start_time - context_offset)
            end_times.append(current_end_time - context_offset)  # Ensure the last word's end time is recorded

        return words, list(zip(start_times, end_times))

    def convert_ids_to_text(self, ids: List[int]) -> List[str]:
        """
        Convert a list of token ids to text.
        
        Args:
            ids (List[int]): The list of token ids.
            
        Returns:
            tokens (List[str]): The list of tokens.
        """
        tokens = []
        for id_ in ids:
            try:
                tokens.append(
                    self.asr_model.tokenizer.ids_to_tokens([int(id_)])[0])
            except KeyError:
                # Handle special or unknown tokens
                tokens.append('<UNK>')

        return tokens

    def get_time_stamps(self, time_stamps: List[int]) -> List[Tuple[float, float]]:
        """
        Get the time stamps for each prediction.
        
        Args:
            time_stamps (List[int]): The list of time stamps for each prediction.
            
        Returns:
            time_stamps (List[Tuple[float, float]]): The list of time stamps for each prediction.
        """
        start_times = time_stamps
        end_times = [t + self.model_stride_in_sec for t in start_times]

        return list(zip(start_times, end_times))


def speech_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate batch of audio signals into a single tensor with their lengths.
    
    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): The batch of audio signals.
        
    Returns:
        audio_signal (torch.Tensor): The batch of audio signals.
        audio_lengths (torch.Tensor): The lengths of the audio signals.
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
