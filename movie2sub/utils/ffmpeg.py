import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def check_ffmpeg() -> None:
    """Check if FFmpeg and FFprobe are available in the system PATH.

    Raises
    ------
    RuntimeError
        If `ffmpeg` or `ffprobe` is not found in the system PATH.
    """

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")

    logger.info("ffmpeg path: %s", ffmpeg or "not found")
    logger.info("ffprobe path: %s", ffprobe or "not found")

    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg executable not found in PATH. "
            "Please install FFmpeg and ensure it's accessible from the command line."
        )

    if ffprobe is None:
        raise RuntimeError(
            "FFprobe executable not found in PATH. "
            "It is typically included with FFmpeg. Make sure it is installed and available in your PATH."
        )


def extract_subtitles(file: str | Path, output_dir_name: str = "subs") -> None:
    """Extract English subtitle streams from a video file using FFmpeg and save them as .srt files.

    Parameters
    ----------
    file : str or Path
        Path to the input video file (e.g., .mkv).
    output_dir_name : str, optional
        Name of the directory to save extracted subtitle files. Default is 'subs'.

    Returns
    -------
    None
    """

    os.makedirs(output_dir_name, exist_ok=True)

    logger.info(f"Output directory ensured at: {output_dir_name}")
    logger.info(f"Scanning subtitle streams in: {file}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "s",
        "-show_entries",
        "stream=index:stream_tags=language,title",
        "-of",
        "json",
        file,
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        logger.error(f"ffprobe failed: {result.stderr.strip()}")
        return

    try:
        streams = json.loads(result.stdout).get("streams", [])
    except json.JSONDecodeError:
        logger.error("Failed to parse ffprobe output.")
        return

    if not streams:
        logger.info("No subtitle streams found.")
        return

    english_streams = [s for s in streams if s.get("tags", {}).get("language", "").lower() in ["eng", "en"]]

    if not english_streams:
        logger.info("Subtitle streams found, but none in English.")
        return

    def subtitle_score(subtitle):
        title = subtitle.get("tags", {}).get("title", "").lower()

        if "sdh" in title:
            return 1

        if "forced" in title:
            return 0

        return 2

    best_stream = max(english_streams, key=subtitle_score)
    index = best_stream["index"]
    lang = best_stream.get("tags", {}).get("language", f"und_{index}")
    title = best_stream.get("tags", {}).get("title", "unknown")
    output_path = os.path.join(output_dir_name, f"subtitle_{lang}_{index}.srt")

    logger.info(f"Extracting subtitle stream {index} (language: {lang}, title: {title}) to {output_path}")

    cmd = ["ffmpeg", "-y", "-i", file, "-map", f"0:{index}", output_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        logger.error(f"Failed to extract subtitle {index}: {result.stderr.strip()}")
    else:
        logger.info(f"Successfully extracted subtitle: {output_path}")
