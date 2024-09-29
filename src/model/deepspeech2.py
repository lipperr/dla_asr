from torch import nn
from torch.nn import Sequential


class DeepSpeech2(nn.Module):
    """
    Deep Speech 2 from http://proceedings.mlr.press/v48/amodei16.pdf
    """

    def __init__(
        self,
        n_features: int,
        n_class: int,
        conv_dim: int = 2,
        n_conv_layers: int = 2,
        n_rnn_layers: int = 5,
        fc_hidden: int = 512,
        **batch,
    ):
        super().__init__()

        self.conv_layers = Sequential(
            nn.Conv2d(
                in_channels=n_features,
                out_channels=32,
                kernel_size=(41, 11),
                stride=(2, 2),
                padding=(20, 5),
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.Hardtanh(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(21, 11),
                stride=(2, 1),
                padding=(10, 5),
                bias=False,
            ),
            nn.BatchNorm2d(num_features=32),
            nn.Hardtanh(),
        )
        rnn_input_size = 16  # (32-41+2*20)//2 + 1
        rnn_output_size = fc_hidden * 2

        self.gru_layers = [  # nn.BatchNorm1d(rnn_input_size),
            nn.GRU(
                input_size=rnn_input_size,
                hidden_size=fc_hidden,
                num_layers=n_rnn_layers,
                bias=True,
                batch_first=True,
                dropout=0.1,
                bidirectional=True,
            )
        ]

        for _ in range(n_rnn_layers - 1):
            # self.gru_layers.append(nn.BatchNorm1d(rnn_output_size)),
            self.gru_layers.append(
                nn.GRU(
                    input_size=rnn_output_size,
                    hidden_size=fc_hidden,
                    num_layers=n_rnn_layers,
                    bias=True,
                    batch_first=True,
                    dropout=0.1,
                    bidirectional=True,
                )
            )
        self.fc = Sequential(
            nn.LayerNorm(rnn_output_size),
            nn.Linear(rnn_output_size, n_class, bias=False),
        )

    def forward(self, spectrogram, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        spectrogram_length = batch["spectrogram_length"]
        outputs, output_lengths = self.conv(spectrogram, spectrogram_length)

        outputs = outputs.permute(1, 0, 2).contiguous()

        for gru_layer in self.gru_layers:
            outputs = gru_layer(outputs, output_lengths)

        outputs = self.fc(outputs.transpose(0, 1)).log_softmax(dim=-1)

        return outputs, output_lengths

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return input_lengths // 2

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
