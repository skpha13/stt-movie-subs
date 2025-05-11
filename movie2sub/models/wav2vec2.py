import os
import warnings
from functools import partial
from typing import Dict

import matplotlib.pyplot as plt
import torch
import torchaudio
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from jiwer import wer
from movie2sub.config import Config, get_project_root, resolve_path
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

    input = processor(waveform.squeeze().numpy(), sampling_rate=resample_rate, return_attention_mask=True)
    input_values = input.input_values[0]

    input_values_min, input_values_max = min(input_values), max(input_values)
    input_values = 2 * (input_values - input_values_min) / (input_values_max - input_values_min) - 1

    batch["input_values"] = input_values
    batch["attention_mask"] = input.attention_mask[0]

    # process text before tokenization
    processed_text = batch["text"].upper().replace(" ", "|").replace("\n", "|")
    batch["labels"] = processor.tokenizer(processed_text).input_ids

    return batch


def prepare_dataset(raw_dataset: Dataset, processor: Wav2Vec2Processor, test_size: float | None = None) -> DatasetDict:
    """Splits a raw dataset and applies preprocessing.

    Parameters
    ----------
    raw_dataset : Dataset
        The raw Hugging Face `Dataset` object.
    processor : Wav2Vec2Processor
        A processor for feature extraction and tokenization.
    test_size: float | None, Optional
        Ratio to split the dataset in train/test sets. Default is None

    Returns
    -------
    DatasetDict
        A dictionary with train and test splits, each containing processed inputs.
    """

    dataset = raw_dataset.train_test_split(test_size=test_size) if test_size is not None else raw_dataset
    preprocess_fn = partial(preprocess, processor=processor)
    return dataset.map(preprocess_fn, remove_columns=["audio"])


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


def inference(model: torch.nn.Module, processor: Wav2Vec2Processor, test_loader: DataLoader):
    """Runs inference on a single batch and prints decoded output.

    Parameters
    ----------
    model : torch.nn.Module
        A trained speech model.
    processor: Wav2Vec2Processor
        The pretrained Wav2Vec2Processor.
    test_loader : DataLoader
        A PyTorch DataLoader containing test data.

    Returns
    -------
    None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for _, batch in enumerate(test_loader):
        with torch.no_grad():
            output = model(input_values=batch["input_values"].to(device))

            with processor.as_target_processor():
                ground_truths = processor.batch_decode(batch["labels"], group_tokens=False, skip_special_tokens=True)

            predicted_ids = torch.argmax(output.logits, dim=-1)
            decoded_texts = processor.batch_decode(predicted_ids, skip_special_tokens=True)

            for j, (gt, pred) in enumerate(zip(ground_truths, decoded_texts)):
                print(f"\tInput {j:02}")
                print(f"Ground truth: {gt}\n")
                print(f"Decoded text: {pred}\n")


training_args = TrainingArguments(
    output_dir="./wav2vec2-finetuned-moviesubs",
    group_by_length=True,
    dataloader_num_workers=4,
    per_device_train_batch_size=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    num_train_epochs=80,
    fp16=False,
    learning_rate=1e-7,
    logging_strategy="steps",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    save_total_limit=1,
    remove_unused_columns=False,
    max_grad_norm=0.05,
)


def train():
    # warnings.filterwarnings("ignore")
    os.environ["WANDB_DISABLED"] = "true"
    torch.autograd.set_detect_anomaly(True)  # to crash in case of anomaly
    model_str = "facebook/wav2vec2-base-960h"

    processor = Wav2Vec2Processor.from_pretrained(
        model_str,
        do_normalize=True,
        feature_size=1,
        padding_value=0.0,
        return_attention_mask=True,
    )
    data_collator = DataCollatorCTC(processor=processor)

    raw_dataset = load_movie2sub_data(os.path.join(Config.get("DATASET_DIR_PATH"), "train"))
    test_dataset = load_movie2sub_data(os.path.join(Config.get("DATASET_DIR_PATH"), "test"))

    dataset = prepare_dataset(raw_dataset, processor, test_size=0.2)
    test_dataset = prepare_dataset(test_dataset, processor)

    model = Wav2Vec2ForCTC.from_pretrained(
        model_str,
        vocab_size=len(processor.tokenizer),
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        ctc_zero_infinity=True,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    trainer.train()

    test_loader = DataLoader(
        test_dataset["test"],
        batch_size=8,
        shuffle=False,
        collate_fn=data_collator,
    )
    inference(model, processor, test_loader)


def transcribe(batch, *, model, processor):  # force model/processor to be keyword-only
    input_values = torch.tensor(batch["input_values"]).unsqueeze(0)
    attention_mask = torch.tensor(batch["attention_mask"]).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

    return {"transcription": transcription}


def test_wav2vec2(checkpoint_dir: str):
    processor = Wav2Vec2Processor.from_pretrained(checkpoint_dir)
    model = Wav2Vec2ForCTC.from_pretrained(checkpoint_dir)
    model.eval()

    test_dataset = load_movie2sub_data(os.path.join(Config.get("DATASET_DIR_PATH"), "test"))
    test_dataset = prepare_dataset(test_dataset, processor)

    transcribe_fn = partial(transcribe, model=model, processor=processor)
    results = test_dataset.map(transcribe_fn)

    for i in range(5):
        reference_str = test_dataset[i]["text"].replace("\n", " ")
        predicted_str = results[i]["transcription"].lower()

        print("Reference:", reference_str)
        print("Predicted:", predicted_str)
        print()


if __name__ == "__main__":
    load_dotenv()
    Config.update_config()

    project_root = get_project_root()
    checkpoints_path = resolve_path("checkpoints/wav2vec2/checkpoint_01", project_root)

    test_wav2vec2(checkpoints_path)
