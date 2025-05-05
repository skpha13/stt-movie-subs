import logging

from dotenv import load_dotenv
from movie2sub.config import Config
from movie2sub.utils.ffmpeg_functions import check_ffmpeg, extract_audio

logging.basicConfig(level=logging.INFO)


def main():
    check_ffmpeg()

    load_dotenv(dotenv_path="../../.env")
    Config.update_config()

    extract_audio(
        Config.get("SAMPLE_MKV_FILE"),
        Config.get("AUDIO_DIR_PATH"),
    )


if __name__ == "__main__":
    main()
