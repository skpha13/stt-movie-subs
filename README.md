# stt-movie-subs

## User Documentation

### Installation

To get started, install the package by following these steps:

```bash
# clone repository
git clone https://github.com/skpha13/stt-movie-subs.git

# enter the directory 
cd stt-movie-subs

# install ffmpeg on linux
sudo apt update
sudo apt install ffmpeg

# install ffmpeg on windows
./scripts/ffmpeg_install.ps1

# install all required dependencies
pip install -e .
```

## Developer Documentation

### Install Optional Packages

For development purposes, install the optional packages with:

```bash
pip install movie2sub[dev]
```

This will install the following tools:

- **black**: A code formatter to ensure consistent style.
- **isort**: A tool for sorting imports automatically.
- **pytest**: A testing framework for running unit tests.

### Running Tests

Run the test suite to ensure everything is working:

```bash
python -m pytest
```
