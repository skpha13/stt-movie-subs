import os
from typing import Dict, Literal, get_args

from dotenv import load_dotenv


def get_project_root() -> str:
    """Find the root directory of the project by walking up the directory tree."""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, ".git")) or os.path.exists(os.path.join(current_dir, "README.md")):
            return current_dir

        current_dir = os.path.dirname(current_dir)

    raise RuntimeError("Project root not found. Make sure your project has a '.git' folder or 'README.md'.")


def is_valid_path(path_str: str) -> bool:
    path_sep = ["/", "\\", r"\\"]

    for sep in path_sep:
        if sep in path_str:
            return True

    return False


def resolve_path(path: str, project_root: str) -> str:
    """Resolves a path to an absolute path if it's relative."""
    if is_valid_path(path) and not os.path.isabs(path):
        return os.path.join(project_root, path)

    return path


CONFIG_KEYS = Literal[
    "SUBTITLE_DIR_PATH",
    "TORRENT_DIR_PATH",
    "DOWNLOAD_DIR_PATH",
    "SAMPLE_MKV_FILE",
    "QBITTORRENT_HOST",
    "QBITTORRENT_PORT",
    "QBITTORRENT_USERNAME",
    "QBITTORRENT_PASSWORD",
]


def create_config_vars(keys: type[Literal]) -> Dict[str, str]:
    keys = list(get_args(keys))
    config_vars = {key: os.getenv(key) for key in keys}

    return config_vars


_config_vars = create_config_vars(CONFIG_KEYS)


class Config:
    _config_vars = _config_vars
    _config_key = CONFIG_KEYS

    @classmethod
    def _resolve_all_paths(cls) -> None:
        """Resolve paths for all config variables."""
        project_root = get_project_root()

        for key in cls._config_vars:
            value = cls._config_vars[key] or os.getenv(key)

            if not value:
                raise RuntimeError(
                    f"Missing environment variable: '{key}'. " "Please set it in your environment or .env file."
                )

            cls._config_vars[key] = resolve_path(value, project_root)

    @classmethod
    def get(cls, key: _config_key) -> str:
        """Get a config variable."""
        return cls._config_vars.get(key)

    @classmethod
    def update_config(cls) -> None:
        """Update all config variables."""
        cls._resolve_all_paths()


if __name__ == "__main__":
    load_dotenv(dotenv_path="../.env")
    Config.update_config()

    keys = list(get_args(CONFIG_KEYS))
    for key in keys:
        print(f"{key}='{Config.get(key)}'")
