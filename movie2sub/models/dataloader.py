import os
import random
from typing import List, Tuple

import torchaudio
from dotenv import load_dotenv
from movie2sub.config import Config
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from movie2sub.utils.preprocessing import AudioProcessor


class MovieSubDataset(Dataset):
    def __init__(
        self, samples: List[Tuple[str, str]], transform=None, target_transform=None, show_progress: bool = True
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.data = []

        iterator = tqdm(samples, desc="Loading samples") if show_progress else samples

        for wav_path, txt_path in iterator:
            waveform, sample_rate = torchaudio.load(wav_path)
            with open(txt_path, "r", encoding="utf-8") as f:
                subtitle_text = f.read()

            features = AudioProcessor.preprocess(waveform, sample_rate)

            if self.transform:
                features = self.transform(features)
            if self.target_transform:
                subtitle_text = self.target_transform(subtitle_text)

            self.data.append((features, subtitle_text))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_movie_subs(root_dir: str, batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load movie subs dataset, split into train/val/test sets, and return corresponding DataLoaders.

    Parameters
    ----------
    root_dir : str
        Path to the dataset directory containing subdirectories for each movie.
    batch_size : int, optional
        Batch size for the DataLoaders. Default is 64.

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        DataLoaders for training, validation, and test datasets.
    """
    all_samples = []

    for movie_name in os.listdir(root_dir):
        movie_path = os.path.join(root_dir, movie_name)
        if not os.path.isdir(movie_path):
            continue

        for fname in os.listdir(movie_path):
            if fname.endswith(".wav"):
                base = os.path.splitext(fname)[0]

                wav_path = os.path.join(movie_path, f"{base}.wav")
                txt_path = os.path.join(movie_path, f"{base}.txt")

                if os.path.exists(wav_path) and os.path.exists(txt_path):
                    all_samples.append((wav_path, txt_path))

    # shuffle and split
    random.seed(42)
    random.shuffle(all_samples)

    train_data, temp_data = train_test_split(all_samples, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # create DataLoaders
    train_loader = DataLoader(MovieSubDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MovieSubDataset(val_data), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(MovieSubDataset(test_data), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def main():
    load_dotenv()
    Config.update_config()

    train_loader, validation_loader, test_loader = load_movie_subs(Config.get("DATASET_DIR_PATH"))

    batch = next(iter(train_loader))
    feature, subtitle = batch[0][0], batch[1][0]

    print("Subtitle text:\n", subtitle)
    print(f"Feature shape: {feature.numpy().shape}")


if __name__ == "__main__":
    main()
