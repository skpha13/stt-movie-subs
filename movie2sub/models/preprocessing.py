import os
from abc import ABC, abstractmethod
from typing import Dict

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchaudio
from dotenv import load_dotenv
from movie2sub.config import Config
from transformers import BertModel, BertTokenizer


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

    This class uses a pre-trained BERT model to tokenize input text and extract contextual embeddings.

    Parameters
    ----------
    tokenizer : str, optional
        The name of the BERT tokenizer to use (default is "bert-base-uncased").
    model : str, optional
        The name of the BERT model to use (default is "bert-base-uncased").
    """

    def __init__(self, tokenizer: str = "bert-base-uncased", model: str = "bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.model = BertModel.from_pretrained(model).to(BertProcessor.device)

    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
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
        return self.tokenizer(text, return_tensors="pt").to(BertProcessor.device)

    def embed_tokens(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate token embeddings from tokenized input using BERT.

        Parameters
        ----------
        tokens : Dict[str, torch.Tensor]
            A dictionary containing tokenized inputs as returned by the tokenizer.

        Returns
        -------
        torch.Tensor
            A tensor of shape (batch_size, sequence_length, hidden_size)
            representing contextual embeddings for each token.
        """
        with torch.no_grad():
            outputs = self.model(**tokens)
            return outputs.last_hidden_state

    def process(self, text: str) -> torch.Tensor:
        """Process the input text by tokenizing and generating BERT embeddings.

        Parameters
        ----------
        text : str
            The input text to process.

        Returns
        -------
        torch.Tensor
            A tensor of contextual embeddings for the input text.
        """
        tokens = self.tokenize_text(text)
        embeddings = self.embed_tokens(tokens)

        return embeddings


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

    features, subtitles, embeddings = zip(*batch)

    # pad embeddings to max sequence length
    max_len = max(e.shape[1] for e in embeddings)
    padded_embeddings = torch.stack([F.pad(e, (0, 0, 0, max_len - e.shape[1])) for e in embeddings])

    features = torch.stack(features)
    return features, list(subtitles), padded_embeddings


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

    processor = BertProcessor()

    with open(os.path.join(Config.get("DATASET_DIR_PATH"), "Django Unchained (2012)", "segment_000003.txt")) as file:
        text = file.read()

    tokens_tensor = processor.tokenize_text(text)
    token_embeddings = processor.embed_tokens(tokens_tensor)

    # convert token IDs to tokens
    input_ids = tokens_tensor["input_ids"][0]
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)

    print(f"Input text: {text}\n")
    print("Token\t\tEmbedding (first 5 values):")
    print("=" * 50)
    for token, embedding in zip(tokens, token_embeddings[0]):
        # Display first 5 dimensions for readability
        embedding_preview = embedding[:5].tolist()
        print(f"{token:<12} {embedding_preview}")
