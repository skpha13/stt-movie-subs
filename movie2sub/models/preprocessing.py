import os
from abc import ABC, abstractmethod
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchaudio
from dotenv import load_dotenv
from movie2sub.config import Config
from transformers import BertTokenizer, Wav2Vec2Processor


class AudioProcessor:
    SAMPLE_RATE = 16_000
    N_MELS = 64
    WINDOW_SIZE = 0.02  # 20 ms
    HOP_LENGTH = 0.01  # 10 ms
    N_FFT = 512
    EPSILON = 1e-6
    MIN_FREQUENCY = 85
    MAX_FREQUENCY = 3000

    # wiener filter parameters
    WIENER_N_FFT = 512
    WIENER_HOP_LENGTH = 128
    WIENER_WIN_LENGTH = 512
    NOISE_FRAME_COUNT = 5  # first 5 frames for noise estimation

    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        win_length=int(WINDOW_SIZE * SAMPLE_RATE),
        hop_length=int(HOP_LENGTH * SAMPLE_RATE),
        n_mels=N_MELS,
        center=True,
        power=2.0,
        f_min=MIN_FREQUENCY,
        f_max=MAX_FREQUENCY,
    )
    log_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    @staticmethod
    def wiener_filter(waveform: torch.Tensor) -> torch.Tensor:
        """Applies Wiener filtering for noise reduction.

        Parameters
        ----------
            waveform: Input audio tensor (1, T)
        Returns
        -------
            Denoised waveform (1, T)
        """
        stft = torch.stft(
            waveform.squeeze(0),
            n_fft=AudioProcessor.WIENER_N_FFT,
            hop_length=AudioProcessor.WIENER_HOP_LENGTH,
            win_length=AudioProcessor.WIENER_WIN_LENGTH,
            window=torch.hann_window(AudioProcessor.WIENER_WIN_LENGTH).to(waveform.device),
            return_complex=True,
        )

        # estimate noise from first few frames
        magnitude = torch.abs(stft)
        noise_estimate = magnitude[:, : AudioProcessor.NOISE_FRAME_COUNT].mean(dim=1, keepdim=True)

        # wiener gain
        gain = (magnitude - noise_estimate).clamp(min=0) / (magnitude + AudioProcessor.EPSILON)

        # reconstruct waveform
        enhanced_stft = stft * gain
        enhanced_waveform = torch.istft(
            enhanced_stft,
            n_fft=AudioProcessor.WIENER_N_FFT,
            hop_length=AudioProcessor.WIENER_HOP_LENGTH,
            win_length=AudioProcessor.WIENER_WIN_LENGTH,
            window=torch.hann_window(AudioProcessor.WIENER_WIN_LENGTH).to(waveform.device),
        )

        return enhanced_waveform.unsqueeze(0)

    @staticmethod
    def preprocess(waveform: torch.Tensor, original_sample_rate: int, apply_wiener: bool = True) -> torch.Tensor:
        """Preprocess the input audio waveform for feature extraction.

        This method performs the following steps:
        1. Converts stereo waveform to mono by averaging across channels.
        2. Normalizes the waveform to the range [-1, 1].
        3. Resamples the audio waveform to a target sample rate (16 kHz) if needed.
        4. Applies a Wiener filter to the waveform, if specified.
        5. Computes a log-Mel spectrogram of the waveform.

        Parameters
        ----------
        waveform : torch.Tensor
            A 1D or 2D tensor representing the audio waveform. Shape: (channels, samples).

        original_sample_rate : int
            The original sample rate of the waveform.

        apply_wiener : bool, optional, default=True
            Whether to apply the Wiener filter to the waveform. If set to False, the Wiener filter is skipped.

        Returns
        -------
        torch.Tensor
            A tensor representing the log-Mel spectrogram of the input waveform. Shape: (n_mel_bins, time_steps).
        """

        # convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # normalize waveform to [-1, 1]
        waveform = waveform / waveform.abs().max()

        # resample to 16kHz if needed
        if original_sample_rate != AudioProcessor.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sample_rate, new_freq=AudioProcessor.SAMPLE_RATE
            )
            waveform = resampler(waveform)

        if apply_wiener:
            waveform = AudioProcessor.wiener_filter(waveform)

        # compute log-mel spectrogram
        mel_spec = AudioProcessor.mel_spectrogram_transform(waveform)
        log_mel_spec = AudioProcessor.log_transform(mel_spec + AudioProcessor.EPSILON)  # add epsilon to avoid log(0)

        # normalize
        mean = log_mel_spec.mean(dim=(1, 2), keepdim=True)
        std = log_mel_spec.std(dim=(1, 2), keepdim=True)
        log_mel_spec = (log_mel_spec - mean) / (std + 1e-5)

        return log_mel_spec


class TextProcessor(ABC):
    """Abstract base class for text processing to generate embeddings."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def process(self, text: str) -> torch.Tensor:
        """Process the input text and return its embedding.

        Parameters
        ----------
        text : str
            The input text to process.

        Returns
        -------
        torch.Tensor
            A tensor representing the processed text embedding.
        """
        pass


class BertProcessor(TextProcessor):
    """BERT-based implementation of the TextProcessor.

    This class uses a pre-trained BERT model to tokenize input text.

    Parameters
    ----------
    tokenizer : str, optional
        The name of the BERT tokenizer to use (default is "bert-base-uncased").
    """

    def __init__(self, tokenizer: str = "bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)

    def tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize input text using the BERT tokenizer.

        Parameters
        ----------
        text : str
            The input string to tokenize.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing input IDs, attention masks, and token type IDs.
        """
        return self.tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0).to(BertProcessor.device)

    def process(self, text: str) -> torch.Tensor:
        """Process the input text by tokenizing and generating BERT embeddings.

        Parameters
        ----------
        text : str
            The input text to process.

        Returns
        -------
        torch.Tensor
            A tensor of token ids for the input text. The token ids maximum values is 30.522.
        """
        token_ids = self.tokenize_text(text)

        return token_ids

    def __len__(self):
        """Get the size of the tokenizer vocabulary.

        Returns
        -------
        int
            The number of tokens in the vocabulary.
        """
        return self.tokenizer.total_vocab_size


class Wav2Vec2TextProcessor(TextProcessor):
    """Wav2Vec2-based processor that extracts token ids from text.

    Parameters
    ----------
    model_name : str, optional
        The pretrained Wav2Vec2 processor to use (default is "facebook/wav2vec2-base-960h").
    """

    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.vocab_dict = self.processor.tokenizer.get_vocab()
        self.char_to_id = {k: v for k, v in self.vocab_dict.items()}
        self.id_to_char = {v: k for k, v in self.vocab_dict.items()}

    def process(self, text: str) -> torch.Tensor:
        """Character based tokenizer

        Parameters
        ----------
        text : str
            The input text to tokenize.

        Returns
        -------
        torch.Tensor
            A tensor of token ids. The token ids maximum values is 32.
        """
        # replace spaces/newlines with the vocab word boundary character
        text = text.upper().replace(" ", "|").replace("\n", "|")
        token_ids = [self.char_to_id.get(char, self.char_to_id["<unk>"]) for char in text]
        return torch.tensor(token_ids)

    def decode_ids_to_tokens(self, token_ids: torch.Tensor) -> List[str]:
        """Convert token IDs back to their corresponding character tokens.

        Parameters
        ----------
        token_ids : torch.Tensor
            A tensor of token IDs.

        Returns
        -------
        List[str]
            A list of character tokens corresponding to the input token IDs.
        """
        return [self.id_to_char.get(token_id.item(), "<unk>") for token_id in token_ids]

    def __len__(self):
        """Get the size of the tokenizer vocabulary.

        Returns
        -------
        int
            The number of tokens in the vocabulary.
        """
        return len(self.vocab_dict)


def collate_fn(batch):
    """Collate function for batching data samples.

    Pads the text embeddings to equal size and stacks the features.

    Parameters
    ----------
    batch : list of tuples
        Each item is a tuple (features, subtitle, embedding), where:
            - features (Tensor): Feature tensor for a sample.
            - subtitle (str): Subtitle text.
            - embedding (Tensor): Text embedding of shape (C, L).

    Returns
    -------
    features : Tensor
        Stacked feature tensors.

    subtitles : list of str
        Subtitle strings for each sample.

    padded_embeddings : Tensor
        Embeddings padded to the same length, shape (B, C, max_len).
    """

    features, subtitles, token_ids = zip(*batch)

    # pad token_ids for CTC loss
    max_token_len = max(len(ids) for ids in token_ids)

    # Create padded tensor and lengths tensor for CTC
    padded_token_ids = torch.zeros(len(token_ids), max_token_len, dtype=torch.long)
    token_lengths = torch.tensor([len(ids) for ids in token_ids], dtype=torch.long)

    for i, ids in enumerate(token_ids):
        padded_token_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    features = torch.stack(features)
    return features, list(subtitles), padded_token_ids, token_lengths


if __name__ == "__main__":
    load_dotenv()
    Config.update_config()

    waveform, sample_rate = torchaudio.load(
        os.path.join(Config.get("DATASET_DIR_PATH"), "Django Unchained (2012)/segment_000003.wav")
    )

    log_mel_spec = AudioProcessor.preprocess(waveform, sample_rate)

    plt.figure(figsize=(10, 6))
    plt.imshow(log_mel_spec.squeeze().numpy(), cmap="viridis", aspect="auto", origin="lower")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Log-Mel Spectrogram")
    plt.xlabel("Frames")
    plt.ylabel("Mel Frequency Bins")
    plt.show()

    processor_bert = BertProcessor()
    processor_wav2vec2 = Wav2Vec2TextProcessor()

    with open(os.path.join(Config.get("DATASET_DIR_PATH"), "Django Unchained (2012)", "segment_000003.txt")) as file:
        text = file.read()

    token_ids_bert = processor_bert.process(text)
    tokens = processor_bert.tokenizer.convert_ids_to_tokens(token_ids_bert)  # convert token ids to tokens

    print(f"Input text: {text}\n")
    print(f"\tBertProcessor")
    print(f"Vocab Size: {len(processor_bert)}")
    print("Token\t\tToken ID:")
    print("=" * 50)
    for token, token_id in zip(tokens, token_ids_bert):
        print(f"{token:<12} {token_id}")

    token_ids_wav2vec2 = processor_wav2vec2.process(text)
    tokens = processor_wav2vec2.decode_ids_to_tokens(token_ids_wav2vec2)

    print(f"\n\tWav2Vec2 Processor")
    print(f"Vocab Size: {len(processor_wav2vec2)}")
    print("Token\t\tToken ID:")
    print("=" * 50)
    for token, token_id in zip(tokens, token_ids_wav2vec2):
        print(f"{token:<12} {token_id.item()}")
