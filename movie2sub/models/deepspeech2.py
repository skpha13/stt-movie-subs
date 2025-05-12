import torch
import torch.nn as nn
from dotenv import load_dotenv
from jiwer import cer, wer
from movie2sub.config import Config
from movie2sub.models.dataloader import load_movie_subs
from movie2sub.models.preprocessing import Wav2Vec2TextProcessor
from torch.utils.data import DataLoader


class FasterDeepSpeech2(nn.Module):
    def __init__(self, num_classes=32):
        """
        Initializes the FasterDeepSpeech2 model with reduced complexity for faster computation.

        Parameters
        ----------
        num_classes : int, optional
            Number of output classes for the final layer. Default is 32.
        """
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, (11, 21), stride=(2, 2), padding=(5, 10)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (11, 11), stride=(1, 1), padding=(5, 5)),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(input_size=32 * 101, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the FasterDeepSpeech2 model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [batch, 1, freq, time].

        Returns
        -------
        torch.Tensor
            Output tensor with shape [batch, time, num_classes].
        """
        x = self.conv_layers(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)

        if x.size(-1) != 3232:
            self.proj = nn.Linear(x.size(-1), 3232).to(x.device)
            x = self.proj(x)

        x, _ = self.rnn(x)
        x = self.fc(x)
        return x


def inference(model: nn.Module, testloader: DataLoader, device: torch.device):
    model.to(device)
    model.eval()

    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in testloader:
            features, subtitles, token_ids = batch
            features = features.to(device)

            output = model(features)

            predicted_token_ids = output.argmax(dim=-1)
            predicted_tokens_batch = []
            for i in range(predicted_token_ids.size(0)):
                predicted_tokens = processor.decode_ids_to_tokens(predicted_token_ids[i])
                predicted_tokens_batch.append(predicted_tokens)

            all_predictions.append(predicted_tokens_batch)
            all_references.append(subtitles)

    wers = []
    cers = []
    for references, predictions in zip(all_references, all_predictions):
        for reference, prediction_tokens in zip(references, predictions):
            reference_str = reference.replace("\n", " ").upper()
            filtered_tokens = [token for token in prediction_tokens if token != "<pad>"]
            prediction_str = "".join(filtered_tokens).replace("|", " ")

            sample_wer = wer(reference, prediction_str)
            wers.append(sample_wer)
            sample_cer = cer(reference, prediction_str)
            cers.append(sample_cer)

            print(f"Reference: {reference_str}")
            print(f"Prediction: {prediction_str}")
            print(f"WER: {sample_wer}")
            print(f"CER: {sample_cer}")
            print()

    avg_wer = sum(wers) / len(wers)
    avg_cer = sum(cers) / len(cers)
    print(f"Average WER on test set: {avg_wer:.4f}")
    print(f"Average CER on test set: {avg_cer:.4f}")


if __name__ == "__main__":
    load_dotenv()
    Config().update_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = Wav2Vec2TextProcessor()

    _, _, test_loader = load_movie_subs(Config.get("DATASET_DIR_PATH"), processor)

    checkpoint_path = "../../checkpoints/DeepSpeech2/deepspeech2.pth"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    model = FasterDeepSpeech2()
    model_state_dict = model.state_dict()

    checkpoint_state_dict = checkpoint["model_state_dict"]

    for key in list(checkpoint_state_dict.keys()):
        if "proj" in key:
            del checkpoint_state_dict[key]

    model_state_dict.update(checkpoint_state_dict)
    model.load_state_dict(model_state_dict)

    inference(model, test_loader, device)
