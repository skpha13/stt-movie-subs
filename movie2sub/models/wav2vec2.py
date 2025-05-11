import os
import warnings
from functools import partial
from typing import Dict

import matplotlib.pyplot as plt
import torch
import torchaudio
from datasets import Dataset, DatasetDict
from jiwer import wer
from torch.utils.data import DataLoader
from transformers import (
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)


def load_movie2sub_data(root_dir: str):
    """Loads audio and corresponding subtitle data paths from a directory structure.

    Parameters
    ----------
    root_dir : str
        Path to the root directory containing movie folders with `.wav` and `.txt` files.

    Returns
    -------
    Dataset
        A Hugging Face `Dataset` containing 'audio' and 'text' fields.
    """
    audio_paths = []
    texts = []

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
                    audio_paths.append(wav_path)
                    with open(txt_path, "r", encoding="utf-8") as f:
                        texts.append(f.read().strip())

    return Dataset.from_dict({"audio": audio_paths, "text": texts})


def preprocess(batch: Dict[str, any], processor: Wav2Vec2Processor, resample_rate=16_000) -> Dict[str, any]:
    """Preprocesses an audio-text batch by resampling, normalizing, and tokenizing.

    Parameters
    ----------
    batch : dict
        A dictionary containing 'audio' (file path) and 'text' (transcription).
    processor : Wav2Vec2Processor
        A processor for feature extraction and tokenization.
    resample_rate : int, optional
        The desired sampling rate for audio, by default 16000.

    Returns
    -------
    dict
        A dictionary with added keys: 'input_values', 'attention_mask', and 'labels'.
    """

    waveform, sample_rate = torchaudio.load(batch["audio"])

    # resample
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=resample_rate)
    waveform = resampler(waveform)

    # convert to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # normalize
    waveform = waveform / waveform.abs().max()

    inputs = processor(waveform.squeeze().numpy(), sampling_rate=resample_rate, return_attention_mask=True)
    input_values = inputs.input_values[0]

    batch["input_values"] = input_values
    batch["attention_mask"] = inputs.attention_mask[0]

    # process text before tokenization
    processed_text = batch["text"].upper().replace(" ", "|").replace("\n", "|")
    batch["labels"] = processor.tokenizer(processed_text).input_ids

    return batch


def prepare_dataset(raw_dataset: Dataset, processor: Wav2Vec2Processor) -> DatasetDict:
    """Splits a raw dataset and applies preprocessing.

    Parameters
    ----------
    raw_dataset : Dataset
        The raw Hugging Face `Dataset` object.
    processor : Wav2Vec2Processor
        A processor for feature extraction and tokenization.

    Returns
    -------
    DatasetDict
        A dictionary with train and test splits, each containing processed inputs.
    """

    dataset = raw_dataset.train_test_split(test_size=0.1)
    preprocess_fn = partial(preprocess, processor=processor)
    return dataset.map(preprocess_fn, remove_columns=["audio", "text"])


def wer_metric(pred: EvalPrediction, processor: Wav2Vec2Processor):
    """Computes the Word Error Rate (WER) between predicted and reference transcriptions.

    Parameters
    ----------
    pred : EvalPrediction
        Predictions from a model evaluation loop, with `predictions` and `label_ids`.
    processor : Wav2Vec2Processor
        A processor used for decoding token ids into strings.

    Returns
    -------
    dict
        A dictionary containing the WER score under the key "wer".
    """

    pred_ids = pred.predictions.argmax(-1)

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer_loss = wer(label_str, pred_str)
    return {"wer": wer_loss}


class DataCollatorCTC:
    """Data collator for CTC-based models. Pads inputs and labels for batch processing.

    Parameters
    ----------
    processor : Wav2Vec2Processor
        Processor used for feature extraction and tokenization.
    """

    def __init__(self, processor: Wav2Vec2Processor):
        self.processor = processor

    def __call__(self, features):
        """Collates a batch of features by padding input values and labels.

        Parameters
        ----------
        features : list of dict
            List of feature dictionaries, each with 'input_values' and 'labels'.

        Returns
        -------
        dict
            A batch dictionary with padded input values and labels.
        """

        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # pad inputs
        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")

        # pad labels
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=True, return_tensors="pt")

        # replace padding with -100 to ignore loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch


def plot_train_validation_loss(trainer: Trainer) -> None:
    """Plots training and validation loss over epochs from a Hugging Face Trainer.

    Parameters
    ----------
    trainer : Trainer
        A Hugging Face `Trainer` object with training logs.

    Returns
    -------
    None
    """

    log_history = trainer.state.log_history

    train_loss = []
    train_epochs = []
    eval_loss = []
    eval_epochs = []

    for log in log_history:
        if "loss" in log and "epoch" in log:
            train_loss.append(log["loss"])
            train_epochs.append(log["epoch"])
        if "eval_loss" in log and "epoch" in log:
            eval_loss.append(log["eval_loss"])
            eval_epochs.append(log["epoch"])

    plt.figure(figsize=(10, 5))
    plt.plot(train_epochs, train_loss, label="Training Loss")
    plt.plot(eval_epochs, eval_loss, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("./train_vs_val_loss.png")
    plt.show()


def inference(model: torch.nn.Module, test_loader: DataLoader):
    """Runs inference on a single batch and prints decoded output.

    Parameters
    ----------
    model : torch.nn.Module
        A trained speech model.
    test_loader : DataLoader
        A PyTorch DataLoader containing test data.

    Returns
    -------
    None
    """

    for i, batch in enumerate(test_loader):
        with torch.no_grad():
            output = model(input_values=batch["input_values"].to(device))
            print("Logits has NaNs:", torch.isnan(output.logits).any().item())

            predicted_ids = torch.argmax(output.logits, dim=-1)
            decoded_text = processor.decode(predicted_ids[0])

            print(f"Decoded text: {decoded_text}")
            break


training_args = TrainingArguments(
    output_dir="./wav2vec2-finetuned-moviesubs",
    group_by_length=True,
    dataloader_num_workers=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    num_train_epochs=80,
    fp16=False,
    learning_rate=1e-6,
    logging_strategy="steps",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    save_total_limit=1,
    remove_unused_columns=False,
    max_grad_norm=0.5,
)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    os.environ["WANDB_DISABLED"] = "true"
    model_str = "facebook/wav2vec2-base-960h"

    processor = Wav2Vec2Processor.from_pretrained(
        model_str,
        do_normalize=True,
        feature_size=1,
        padding_value=0.0,
        return_attention_mask=True,
    )
    data_collator = DataCollatorCTC(processor=processor)

    raw_dataset = load_movie2sub_data("/kaggle/input/movie2sub-dataset/dataset")
    dataset = prepare_dataset(raw_dataset, processor)

    model = Wav2Vec2ForCTC.from_pretrained(
        model_str,
        vocab_size=len(processor.tokenizer),
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    wrapped_compute_metrics = partial(wer_metric, processor=processor)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=processor,
        data_collator=data_collator,
        compute_metrics=wrapped_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    trainer.train()

    test_loader = DataLoader(
        dataset["test"],
        batch_size=4,
        shuffle=False,
        collate_fn=data_collator,
    )
    inference(model, test_loader)
