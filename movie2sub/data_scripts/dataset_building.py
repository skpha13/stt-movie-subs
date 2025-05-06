import logging
import os
import shutil

from dotenv import load_dotenv
from movie2sub.config import Config
from movie2sub.utils.audio import export_segments, segment_subtitles
from movie2sub.utils.ffmpeg_functions import extract_audio, extract_subtitles
from movie2sub.utils.files import is_srt_file, is_video_file
from movie2sub.utils.torrent import QBittorrent, QBittorrentConnectionInfo
from movie2sub.utils.yts_scraper import YTSWebScraper

logger = logging.getLogger(__name__)


def scrape_torrents(output_dir_path: str):
    YTSWebScraper.OUTPUT_DIR_PATH = output_dir_path

    scraper = YTSWebScraper("https://yts.mx/browse-movies/Quentin%20Tarantino")
    scraper.scrape_pages()


def download_torrents(download_dir_path: str, torrent_dir_path: str):
    conn_info: QBittorrentConnectionInfo = QBittorrentConnectionInfo(
        Config.get("QBITTORRENT_HOST"),
        int(Config.get("QBITTORRENT_PORT")),
        Config.get("QBITTORRENT_USERNAME"),
        Config.get("QBITTORRENT_PASSWORD"),
    )

    logger.info("Logging in...")
    qb = QBittorrent(conn_info=conn_info)
    qb.client_log_in()
    logger.info("Connected...")

    os.makedirs(download_dir_path, exist_ok=True)

    logger.info("Adding torrents...")
    for filename in os.listdir(torrent_dir_path):
        torrent_path = os.path.join(torrent_dir_path, filename)

        qb.add_torrent(torrent_path, download_dir_path)

    qb.monitor_progress(interval=10)
    logger.info("Download finished.")


def extract_audio_and_subtitles(audio_sub_dir_path: str, download_dir_path: str):
    for movie_dir in os.listdir(download_dir_path):
        movie_dir_path = os.path.join(download_dir_path, movie_dir)
        extraction_path = os.path.join(audio_sub_dir_path, movie_dir)
        os.makedirs(extraction_path, exist_ok=True)

        for filename in os.listdir(movie_dir_path):
            filepath = os.path.join(movie_dir_path, filename)

            if is_video_file(filepath):
                extract_audio(filepath, extraction_path)
                extract_subtitles(filepath, extraction_path)

            elif is_srt_file(filepath):
                shutil.copy(filepath, os.path.join(extraction_path, filename))


def audio_segmenting(audio_sub_dir_path: str, dataset_dir_path: str):
    for movie_name in os.listdir(audio_sub_dir_path):
        movie_folder_path = os.path.join(audio_sub_dir_path, movie_name)

        audio_file_path = None
        sub_file_path = None
        for file_name in os.listdir(movie_folder_path):
            file_path = os.path.join(movie_folder_path, file_name)
            if file_name.endswith(".wav"):
                audio_file_path = file_path
            elif file_name.endswith(".srt"):
                sub_file_path = file_path

        if not audio_file_path or not sub_file_path:
            logger.info(f"Directory: {movie_folder_path} has missing files. Skipping...")
            continue

        output_dir_path = os.path.join(dataset_dir_path, movie_name)
        try:
            segments = segment_subtitles(sub_file_path)
        except Exception as e:
            logger.info(f"Failed to segment subtitle: {sub_file_path}, Error: {e}")
            continue
        export_segments(audio_file_path, segments, output_dir_path)


def main():
    scrape_torrents(TORRENT_DIR_PATH)
    download_torrents(DOWNLOAD_DIR_PATH, TORRENT_DIR_PATH)

    extract_audio_and_subtitles(AUDIO_SUB_DIR_PATH, DOWNLOAD_DIR_PATH)

    audio_segmenting(AUDIO_SUB_DIR_PATH, DATASET_DIR_PATH)


if __name__ == "__main__":
    load_dotenv("../../.env")
    Config.update_config()

    TORRENT_DIR_PATH = Config.get("TORRENT_DIR_PATH")
    DOWNLOAD_DIR_PATH = Config.get("DOWNLOAD_DIR_PATH")
    AUDIO_SUB_DIR_PATH = Config.get("AUDIO_SUB_PATH")
    DATASET_DIR_PATH = Config.get("DATASET_DIR_PATH")

    logging.basicConfig(level=logging.INFO)

    main()
