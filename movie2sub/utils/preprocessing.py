import os

import matplotlib.pyplot as plt
import torch
import torchaudio
from dotenv import load_dotenv
from movie2sub.config import Config


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
