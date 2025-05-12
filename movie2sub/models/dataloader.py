import os
import random
from typing import List, Tuple

import torchaudio
from dotenv import load_dotenv
from movie2sub.config import Config
from movie2sub.models.preprocessing import AudioProcessor, TextProcessor, Wav2Vec2TextProcessor, collate_fn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class MovieSubDataset(Dataset):
    """A Dataset class for loading and processing audio and subtitle pairs for movie-related tasks."""

    def __init__(self, samples: List[Tuple[str, str]], text_processor: TextProcessor, show_progress: bool = True):
        """Initializes the MovieSubDataset by loading and processing the audio and subtitle data.

        Parameters
        ----------
        samples : list of tuple
            A list of tuples, where each tuple contains the file paths to an audio file (wav) and a subtitle file (txt).
        text_processor : TextProcessor
            An instance of the `TextProcessor` class for text processing.
        show_progress : bool, optional, default=True
            If True, displays a progress bar during the loading of samples.
        """

        self.data = []
        self.text_processor = text_processor

        iterator = tqdm(samples, desc="Loading samples") if show_progress else samples

        for wav_path, txt_path in iterator:
            waveform, sample_rate = torchaudio.load(wav_path)
            with open(txt_path, "r", encoding="utf-8") as f:
                subtitle_text = f.read()

            features = AudioProcessor.preprocess(waveform, sample_rate)
            text_embeddings = self.text_processor.process(subtitle_text)

            self.data.append((features, subtitle_text, text_embeddings))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def fetch_samples(root_dir: str) -> List[Tuple[str, str]]:
    """Traverse the directory structure to collect pairs of .wav and .txt file paths.

    Parameters
    ----------
    root_dir : str
       The root directory containing subdirectories, each representing a movie
       with .wav and corresponding .txt transcript files.

    Returns
    -------
    List[Tuple[str, str]]
       A list of tuples, where each tuple contains the full paths to a .wav audio file
       and its corresponding .txt transcript file. Only pairs where both files exist are included.
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

    return all_samples


def load_movie_subs(
    root_dir: str, processor: TextProcessor, batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Loads movie subtitle datasets and returns DataLoader instances for training, validation, and testing.

    Parameters
    ----------
    root_dir : str
       The root directory containing the "train" and "test" subdirectories with audio and subtitle files.
    processor : TextProcessor
       An instance of the `TextProcessor` class used to process subtitle text and generate text embeddings.
    batch_size : int, optional, default=64
       The batch size used for loading the data in the DataLoader.

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
       A tuple containing the following DataLoader instances:
           - train_loader: A DataLoader for the training dataset.
           - val_loader: A DataLoader for the validation dataset.
           - test_loader: A DataLoader for the test dataset.
    """

    all_samples = fetch_samples(os.path.join(root_dir, "train"))
    test_samples = fetch_samples(os.path.join(root_dir, "test"))

    # shuffle and split
    random.seed(42)
    random.shuffle(all_samples)

    train_data, val_data = train_test_split(all_samples, test_size=0.3, random_state=42)

    # create DataLoaders
    train_loader = DataLoader(
        MovieSubDataset(train_data, processor), batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        MovieSubDataset(val_data, processor), batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        MovieSubDataset(test_samples, processor), batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader


def main():
    load_dotenv()
    Config.update_config()

    processor = Wav2Vec2TextProcessor()
    train_loader, validation_loader, test_loader = load_movie_subs(Config.get("DATASET_DIR_PATH"), processor)

    batch = next(iter(train_loader))
    feature, subtitle, token_ids = batch[0][0], batch[1][0], batch[2][0]
    tokens = processor.decode_ids_to_tokens(token_ids)

    print(f"Subtitle: {subtitle}\n")
    print(f"\n\tWav2Vec2 Processor")
    print(f"Vocab Size: {len(processor)}")
    print("Token\t\tToken ID:")
    print("=" * 50)
    for token, token_id in zip(tokens, token_ids):
        print(f"{token:<12} {token_id.item()}")


if __name__ == "__main__":
    main()
