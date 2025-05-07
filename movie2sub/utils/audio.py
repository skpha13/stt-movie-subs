import logging
import os
from datetime import time
from pathlib import Path
from typing import List

import numpy as np
import pysrt
import soundfile as sf
from dotenv import load_dotenv
from movie2sub.config import Config
from movie2sub.utils.custom_types import Segment
from movie2sub.utils.text import clean_asr_text
from tqdm import tqdm

logger = logging.getLogger(__name__)


def time_to_seconds(t: time) -> float:
    """Convert a `time` object to the total number of seconds, including fractional seconds.

    Parameters
    ----------
    t : time
        A `time` object representing a specific time.

    Returns
    -------
    float
        The total number of seconds represented by the `time` object, including fractional seconds
        (microseconds).
    """
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def pad_audio(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pads the input audio array to the specified target length with zeros (silence) at the end.

    Parameters
    ----------
    audio : np.ndarray
        A 1D NumPy array representing the audio data to be padded.

    target_length : int
        The desired length of the output audio in terms of number of samples.

    Returns
    -------
    np.ndarray
        A 1D NumPy array of length `target_length`.
    """

    if audio.shape[0] < target_length:
        padding = target_length - audio.shape[0]
        return np.pad(audio, ((0, padding), (0, 0)), mode="constant")  # pad along each channel

    return audio


class AudioSubSegmenter:
    def __init__(self, audio_file: str | Path, subtitle_file: str | Path, max_segment_length: int = 30):
        self.max_segment_length = max_segment_length

        self.audio_file = audio_file
        self.subtitle_file = subtitle_file

        self.subs = pysrt.open(subtitle_file)
        self.audio, self.sample_rate = sf.read(audio_file)

    def segment_subtitles(self) -> List[Segment]:
        """Segments subtitles into chunks where the duration of each segment does not exceed the specified
        maximum segment length.

        Returns
        -------
        List[Segment]
            A list of `Segment` objects representing the subtitle segments.
        """

        logger.info(f"Processing subtitle file: {self.subtitle_file}")

        segments = []
        current_segment = []
        segment_start_time = self.subs[0].start.to_time()

        for sub in self.subs:
            if not current_segment:
                segment_start_time = sub.start.to_time()

            potential_end_time = sub.end.to_time()
            duration = time_to_seconds(potential_end_time) - time_to_seconds(segment_start_time)

            if duration <= self.max_segment_length:
                current_segment.append(sub)
            else:
                # finalize segment respecting max_segment_length
                last_end_time = current_segment[-1].end.to_time()
                segments.append(Segment(segment_start_time, last_end_time, current_segment))

                # start new one with overflow from last segment
                current_segment = [sub]
                segment_start_time = sub.start.to_time()

        # add any remaining subtitles
        if current_segment:
            segment_end_time = current_segment[-1].end.to_time()
            segments.append(Segment(segment_start_time, segment_end_time, current_segment))

        logger.info(f"Found {len(segments)} segments")

        return segments

    def export_segments(self, segments: List[Segment], output_dir: str | Path) -> None:
        """Exports the segments as audio and corresponding subtitle files.

        Parameters
        ----------
        segments : List[Segment]
            A list of `Segment` objects representing the segments to be exported.

        output_dir : str or Path
            The directory where the exported audio and subtitle files will be saved.

        Returns
        -------
        None
            This function does not return any value. It writes audio and subtitle files to the specified
            output directory.

        Notes
        -----
        Audio is exported as `.wav` files, and subtitles are exported as plain text files for each segment.
        """

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Segmenting audio file: {self.audio_file}")

        for index, segment in tqdm(enumerate(segments), total=len(segments), desc="Processing segments"):
            start_time = segment.start_time
            end_time = segment.end_time
            subs_chunk = segment.subtitle

            start_sec = time_to_seconds(start_time)
            end_sec = time_to_seconds(end_time)
            duration = end_sec - start_sec

            duration_threshold = 10.0
            if duration < duration_threshold:
                logger.info(f"Skipping segment {index:06d}: duration < {duration_threshold} second ({duration:.2f}s)")
                continue

            start_sample = int(start_sec * self.sample_rate)
            end_sample = int(end_sec * self.sample_rate)

            # audio segmentation
            audio_chunk = self.audio[start_sample:end_sample]
            audio_chunk = pad_audio(audio_chunk, self.max_segment_length * self.sample_rate)

            # export subtitles as plain text
            txt_path = os.path.join(output_dir, f"segment_{index:06d}.txt")
            clean_subs = clean_asr_text("\n".join(map(lambda sub: sub.text, subs_chunk)))
            word_count = len(clean_subs.strip().split())
            word_count_threshold = 15

            if word_count < word_count_threshold:
                logger.info(
                    f"Skipping segment {index:06d}: subtitle has less than {word_count_threshold} words ({word_count} words)"
                )
                continue

            # export audio
            sf.write(os.path.join(output_dir, f"segment_{index:06d}.wav"), audio_chunk, self.sample_rate)

            # export subtitles
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(clean_subs)

        logger.info("Segmenting successful")


if __name__ == "__main__":
    load_dotenv()
    Config.update_config()

    logging.basicConfig(level=logging.INFO)

    audio_file = os.path.join(Config.get("AUDIO_DIR_PATH"), "Dazed.and.Confused.1993.1080p.BluRay.DTS.x264-VietHD.wav")
    sub_file = os.path.join(Config.get("SUBTITLE_DIR_PATH"), "subtitle_eng_3.srt")
    segmenter = AudioSubSegmenter(audio_file, sub_file, max_segment_length=30)

    # need to have subtitles extracted for the selected audio first
    segments = segmenter.segment_subtitles()

    # need to have this audio file that corresponds with the subtitles
    segmenter.export_segments(segments, "../../data/segmented_audio")
