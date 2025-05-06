import logging
import time
from dataclasses import asdict, dataclass
from typing import List

import qbittorrentapi
from qbittorrentapi.torrents import TorrentStatusesT

logger = logging.getLogger(__name__)


@dataclass
class QBittorrentConnectionInfo:
    host: str
    port: int
    username: str
    password: str


class QBittorrent:
    """Wrapper class for interacting with a qBittorrent Web UI using the qbittorrent-api package.

    Parameters
    ----------
    conn_info: QBittorrentConnectionInfo
        The connection info (host, port, username, password).

    Attributes
    ----------
    qbt_client : Optional[qbittorrentapi.Client]
        The connected qBittorrent client instance.
    conn_info : dict
        Dictionary storing host, port, username, and password for login.
    """

    def __init__(self, conn_info: QBittorrentConnectionInfo) -> None:
        self.qbt_client: qbittorrentapi.client.Client | None = None
        self.host = conn_info.host
        self.port = conn_info.port
        self.username = conn_info.username
        self.password = conn_info.password

        self.conn_info = asdict(conn_info)

    def client_log_in(self):
        """Logs in to the qBittorrent Web UI using stored credentials.

        Raises
        ------
        qbittorrentapi.LoginFailed
            If authentication fails.
        """

        self.qbt_client = qbittorrentapi.Client(**self.conn_info)

        try:
            self.qbt_client.auth_log_in()
        except qbittorrentapi.LoginFailed as e:
            logger.error("Login failed: %s", e)

    def client_log_out(self):
        """Logs out from the qBittorrent Web UI session, if connected."""
        if self.qbt_client:
            try:
                self.qbt_client.auth_log_out()
            except Exception as e:
                logger.error("Logout failed: %s", e)

    def add_torrent(self, file_path: str, save_path: str) -> None:
        """Adds a .torrent file to the qBittorrent client for download.

        Parameters
        ----------
        file_path : str
            The full path to the .torrent file.
        save_path : str
            The directory where downloaded files should be saved.

        Raises
        ------
        Exception
            If adding the torrent fails.
        """
        if not self.qbt_client:
            logger.error("Client not logged in. Cannot add torrent.")
            return

        try:
            self.qbt_client.torrents_add(torrent_files=file_path, save_path=save_path)
        except Exception as e:
            logger.error("Failed to add torrent: %s", e)

    def stop_all_torrents(self) -> None:
        """Stops (pauses) all active torrents in the qBittorrent client.

        Raises
        ------
        Exception
            If pausing the torrents fails.
        """
        if not self.qbt_client:
            logger.error("Client not logged in. Cannot stop torrents.")
            return

        try:
            self.qbt_client.torrents_pause()
            logger.info("All torrents paused successfully.")
        except Exception as e:
            logger.error("Failed to pause torrents: %s", e)

    def remove_all_torrents(self, delete_files=False):
        """Removes all torrents from the qBittorrent client.

        Parameters
        ----------
        delete_files : bool
            If True, also delete the downloaded files associated with the torrents. Default is False.
        """
        try:
            torrents = self.qbt_client.torrents_info()

            if not torrents:
                logger.info("No torrents found to remove.")
                return

            torrent_hashes = [torrent.hash for torrent in torrents]
            self.qbt_client.torrents_delete(delete_files=delete_files, hashes=torrent_hashes)

            logger.info(f"Removed {len(torrent_hashes)} torrents.")
        except Exception as e:
            logger.exception("Failed to remove torrents: %s", e)

    def display_progress(self) -> None:
        """Displays the progress of all active torrents."""
        if not self.qbt_client:
            logger.error("Client not logged in. Cannot display torrent progress.")
            return

        try:
            torrents = self.qbt_client.torrents_info(status_filter="downloading")
            for torrent in torrents:
                progress = torrent.progress * 100
                logger.info(
                    f"Torrent: {torrent.name} | Progress: {progress:.2f}% | "
                    f"Downloaded: {torrent.downloaded} | Size: {torrent.total_size}"
                )
        except Exception as e:
            logger.exception("Failed to fetch torrents progress: %s", e)

    def monitor_progress(self, interval=5) -> None:
        """Continuously checks and displays progress for all active torrents."""
        try:
            while True:
                status_filters: List[TorrentStatusesT] = [
                    "downloading",
                    "checking",
                    "stalled",
                    "stalled_downloading",
                ]

                torrents = []
                for status_filter in status_filters:
                    torrents_filtered = self.qbt_client.torrents_info(status_filter=status_filter)
                    # contributing status also sets stalled status, so we need to filter manually torrents which finished downloading
                    torrents_filtered = list(filter(lambda torrent: torrent.progress < 1, torrents_filtered))

                    torrents.extend(torrents_filtered)

                if not torrents:
                    logger.info("No torrents are currently downloading. Stopping progress monitoring.")
                    break

                self.display_progress()
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Progress monitoring stopped.")
