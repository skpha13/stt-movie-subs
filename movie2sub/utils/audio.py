import logging
import os
from datetime import time
from pathlib import Path
from typing import List

import pysrt
import soundfile as sf
from dotenv import load_dotenv
from movie2sub.config import Config
from movie2sub.utils.custom_types import Segment
from movie2sub.utils.text import clean_asr_text
from tqdm import tqdm

logger = logging.getLogger(__name__)


def time_to_seconds(t: time) -> float:
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def segment_subtitles(subtitle_file: str | Path, max_segment_length: int = 30) -> List[Segment]:
    logger.info(f"Processing subtitle file: {subtitle_file}")
    subs = pysrt.open(subtitle_file)

    segments = []
    current_segment = []
    segment_start_time = subs[0].start.to_time()

    for sub in subs:
        if not current_segment:
            segment_start_time = sub.start.to_time()

        potential_end_time = sub.end.to_time()
        duration = time_to_seconds(potential_end_time) - time_to_seconds(segment_start_time)

        if duration <= max_segment_length:
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


def export_segments(audio_file: str | Path, segments: List[Segment], output_dir: str | Path) -> None:
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Segmenting audio file: {audio_file}")

    audio, sample_rate = sf.read(audio_file)

    for index, segment in tqdm(enumerate(segments), total=len(segments), desc="Processing segments"):
        start_time = segment.start_time
        end_time = segment.end_time
        subs_chunk = segment.subtitle

        start_sec = time_to_seconds(start_time)
        end_sec = time_to_seconds(end_time)

        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)

        # export audio
        audio_chunk = audio[start_sample:end_sample]
        sf.write(os.path.join(output_dir, f"segment_{index:06d}.wav"), audio_chunk, sample_rate)

        # export subtitles as plain text
        txt_path = os.path.join(output_dir, f"segment_{index:06d}.txt")
        clean_subs = clean_asr_text("\n".join(map(lambda sub: sub.text, subs_chunk)))
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(clean_subs)

    logger.info("Segmenting successful")


if __name__ == "__main__":
    load_dotenv()
    Config.update_config()

    logging.basicConfig(level=logging.INFO)

    segments = segment_subtitles(os.path.join(Config.get("SUBTITLE_DIR_PATH"), "subtitle_eng_3.srt"))
    export_segments(
        os.path.join(Config.get("AUDIO_DIR_PATH"), "Dazed.and.Confused.1993.1080p.BluRay.DTS.x264-VietHD.wav"),
        segments,
        "../../data/segmented_audio",
    )
