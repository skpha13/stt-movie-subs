import re


def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def is_video_file(file_path):
    return file_path.lower().endswith((".mp4", ".mkv"))
