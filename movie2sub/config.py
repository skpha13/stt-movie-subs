import os

from dotenv import load_dotenv


def get_project_root() -> str:
    """Find the root directory of the project by walking up the directory tree."""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, ".git")) or os.path.exists(os.path.join(current_dir, "README.md")):
            return current_dir

        current_dir = os.path.dirname(current_dir)

    raise RuntimeError("Project root not found. Make sure your project has a '.git' folder or 'README.md'.")


def resolve_path(path: str, project_root: str) -> str:
    """Resolves a path to an absolute path if it's relative."""
    if path and not os.path.isabs(path):
        return os.path.join(project_root, path)

    return path


class Config:
    _config_vars = {
        "SUBTITLE_DIR_PATH": os.getenv("SUBTITLE_DIR_PATH"),
        "SAMPLE_MKV_FILE": os.getenv("SAMPLE_MKV_FILE"),
    }

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
    def get(cls, key: str) -> str:
        """Get a config variable."""
        return cls._config_vars.get(key)

    @classmethod
    def update_config(cls) -> None:
        """Update all config variables."""
        cls._resolve_all_paths()


if __name__ == "__main__":
    load_dotenv(dotenv_path="../.env.public")
    Config.update_config()

    print(Config.get("SUBTITLE_DIR_PATH"))
    print(Config.get("SAMPLE_MKV_FILE"))
