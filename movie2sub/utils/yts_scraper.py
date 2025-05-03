import logging
import os
import random
import time
from typing import Dict, Tuple

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fake_useragent import UserAgent
from movie2sub.config import Config
from movie2sub.utils.files import sanitize_filename

logger = logging.getLogger(__name__)


class YTSWebScraper:
    DIRECTOR_NAME: str = "quentin tarantino"
    OUTPUT_DIR_PATH: str = Config.get("TORRENT_DIR_PATH")

    def __init__(self, base_url: str, delay_range: Tuple[float, float] = (1.0, 3.0)):
        self.base_url = base_url
        self.ua = UserAgent()
        self.headers: Dict[str, str] = self._get_random_headers()
        self.delay_range = delay_range

    def _get_random_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": self.ua.random,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Connection": "keep-alive",
            "DNT": "1",
        }

    def _get_page_source(self, url: str, params=None) -> BeautifulSoup:
        time.sleep(random.uniform(*self.delay_range))
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()

            self.headers = self._get_random_headers()
            return BeautifulSoup(response.text, "html.parser")
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return BeautifulSoup("", "html.parser")

    def _download_torrents(self, movie_links: list):
        for url in movie_links:
            logger.info(f"\tProcessing URL: {url}")
            soup = self._get_page_source(url)

            # request failed for the current link, skip to next
            if not soup.contents:
                continue

            movie_info = soup.find("div", {"id": "movie-info"})
            movie_title = sanitize_filename(movie_info.find("h1").text.strip())

            director_tag = soup.select_one('span[itemprop="director"] span[itemprop="name"]')
            director_name = director_tag.text.strip() if director_tag else None

            if director_name is not None and director_name.lower() != YTSWebScraper.DIRECTOR_NAME:
                logger.info(
                    f"Skipping movie directed by '{director_name}', does not match target director '{YTSWebScraper.DIRECTOR_NAME}'."
                )
                continue

            # first <a> is usually the lowest quality one (720p)
            torrent_tag = movie_info.find("a")
            torrent_url = torrent_tag["href"]

            try:
                torrent_response = requests.get(torrent_url, headers=self._get_random_headers(), timeout=10)
                torrent_response.raise_for_status()

                file_path = os.path.join(YTSWebScraper.OUTPUT_DIR_PATH, f"{movie_title}.torrent")

                with open(file_path, "wb") as f:
                    f.write(torrent_response.content)

                logger.info(f"Saved: {file_path}")
            except requests.RequestException as e:
                logger.error(f"Failed to download torrent: {e}")

    @staticmethod
    def _get_movie_links_from_page(soup: BeautifulSoup):
        movie_cards = soup.select("div.browse-movie-wrap")
        links = []

        for card in movie_cards:
            a_tag = card.find("a")
            if a_tag:
                movie_link = a_tag["href"]
                links.append(movie_link)

        return links

    def scrape_pages(self):
        page_index = 1
        url = self.base_url

        while True:
            logger.info(f"Processing Page: {page_index}")
            params = {"page": page_index} if page_index != 1 else None
            soup = self._get_page_source(url, params=params)

            # failed request for current page, skip to next
            if not soup.contents:
                page_index += 1
                continue

            movie_links = YTSWebScraper._get_movie_links_from_page(soup)

            # no movie links found => end of pagination
            if not movie_links:
                break

            self._download_torrents(movie_links)

            page_index += 1


if __name__ == "__main__":
    load_dotenv("../../.env")
    Config.update_config()
    YTSWebScraper.OUTPUT_DIR_PATH = Config.get("TORRENT_DIR_PATH")

    logging.basicConfig(level=logging.INFO)

    scraper = YTSWebScraper("https://yts.mx/browse-movies/Quentin%20Tarantino")
    scraper.scrape_pages()
