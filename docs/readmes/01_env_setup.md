# Environment Setup

## Create Your `.env` File

Start by creating your environment configuration file. A sample template is already provided:

```bash
# linux
cp .env.public .env

# windows
copy .env.public .env
```

This copies the `.env.public` file (found at the project root) and renames it to `.env`.

## Configure Environment Variables

Some paths are pre-configured with default values:

```bash
SUBTITLE_DIR_PATH="data\\subs"
TORRENT_DIR_PATH="data\\torrents"
DOWNLOAD_DIR_PATH="data\\downloads"
AUDIO_DIR_PATH="data\\audio"
AUDIO_SUB_PATH="data\\audio_sub"
DATASET_DIR_PATH="data\\dataset"
```
> We recommend to not change these.

However, you’ll need to manually set your qBittorrent credentials:

```bash
QBITTORRENT_HOST="localhost"     # Default qBittorrent Web UI host
QBITTORRENT_PORT=8080            # Default Web UI port
QBITTORRENT_USERNAME=""          # Your qBittorrent username
QBITTORRENT_PASSWORD=""          # Your qBittorrent password
```

> [!NOTE]
> Make sure your qBittorrent Web UI is enabled and configured correctly.

For detailed instructions, refer to:

[**qBittorrent Setup Guide**](./02_qbittorrent_setup.md)