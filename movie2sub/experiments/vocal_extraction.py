import os
import subprocess

from dotenv import load_dotenv
from movie2sub.config import Config
from spleeter.separator import Separator


def spleeter_extraction(input_file: str, output_dir: str) -> None:
    separator = Separator("spleeter:5stems", multiprocess=False)
    separator.separate_to_file(
        input_file, output_dir, filename_format="spleeter_5stems/{filename}/{instrument}.{codec}"
    )


def demucs_extraction(input_file: str, output_dir: str) -> None:
    model = "htdemucs_ft"

    subprocess.run(
        [
            "demucs",
            "-n",
            model,
            "--out",
            output_dir,
            input_file,
        ]
    )


def main():
    output_dir = "../../data/vocal_extraction"
    input_file = Config.get("SAMPLE_WAV_FILE")

    os.makedirs(output_dir, exist_ok=True)

    spleeter_extraction(input_file, output_dir)
    demucs_extraction(input_file, output_dir)


if __name__ == "__main__":
    load_dotenv(dotenv_path="../../.env")
    Config.update_config()

    main()
