import logging
import os

from dotenv import load_dotenv
from movie2sub.config import Config
from movie2sub.utils.torrent import QBittorrent, QBittorrentConnectionInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    load_dotenv()
    Config.update_config()

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

    logger.info("Adding torrents...")
    qb.add_torrent(
        os.path.join(
            Config.get("TORRENT_DIR_PATH"), "Cyberpunk.Edgerunners.S01.1080p.NF.WEB-DL.DDP5.1.H.264-SMURF.torrent"
        ),
        Config.get("DOWNLOAD_DIR_PATH"),
    )

    qb.monitor_progress(interval=10)
    logger.info("Download finished.")


if __name__ == "__main__":
    main()
