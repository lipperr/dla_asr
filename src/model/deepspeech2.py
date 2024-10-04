from torch import nn
from torch.nn import Sequential


class GRUBlock(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        batch_first=True,
        dropout=0.1,
        bidirectional=True,
    ):
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(input_size)
        self.gru_layer = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x, x_length):
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2).contiguous()
        x = nn.utils.rnn.pack_padded_sequence(x, x_length, batch_first=True)
        x = self.gru_layer(x, x_length)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.gru_layer.bidirectional:
            x = (
                x[..., : self.gru_layer.hidden_size]
                + x[..., self.gru_layer.hidden_size :]
            )
        return x


class ConvBlock(nn.Module):
    def __init__(
        self, conv_dim: int = 2, n_conv_layers: int = 2, n_features: int = 128
    ):
        super().__init__()
        assert conv_dim in [1, 2] and n_conv_layers in [1, 2, 3]
        self.scaling = 2
        self.out_features = 0
        if conv_dim == 1:
            if n_conv_layers == 1:
                self.conv_block = nn.Sequential(
                    nn.Conv1d(
                        in_channels=1,
                        out_channels=1280,
                        kernel_size=11,
                        stride=2,
                        padding=5,
                    ),
                    nn.BatchNorm1d(1280),
                    nn.Hardtanh(0, 20, inplace=True),
                )
                self.out_features = (n_features + 5 * 2 - 11) // 2 + 1

            elif n_conv_layers == 2:
                self.conv_block = nn.Sequential(
                    nn.Conv1d(
                        in_channels=1,
                        out_channels=640,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                    ),
                    nn.BatchNorm1d(640),
                    nn.Hardtanh(0, 20, inplace=True),
                    nn.Conv1d(
                        in_channels=640,
                        out_channels=640,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                    ),
                    nn.BatchNorm1d(640),
                    nn.Hardtanh(0, 20, inplace=True),
                )
                self.out_features = (n_features + 2 * 1 - 5) + 1
                self.out_features = (self.out_features + 2 * 2 - 5) // 2 + 1
            else:
                self.conv_block = nn.Sequential(
                    nn.Conv1d(
                        in_channels=1,
                        out_channels=512,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                    ),
                    nn.BatchNorm1d(512),
                    nn.Hardtanh(0, 20, inplace=True),
                    nn.Conv1d(
                        in_channels=512,
                        out_channels=512,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                    ),
                    nn.BatchNorm1d(512),
                    nn.Hardtanh(0, 20, inplace=True),
                    nn.Conv1d(
                        in_channels=512,
                        out_channels=512,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                    ),
                    nn.BatchNorm1d(512),
                    nn.Hardtanh(0, 20, inplace=True),
                )
                self.out_features = (n_features + 2 * 1 - 5) + 1
                self.out_features = (self.out_features + 2 * 1 - 5) + 1
                self.out_features = (self.out_features + 2 * 2 - 5) // 2 + 1
        else:
            if n_conv_layers == 1:
                self.conv_block = nn.Sequential(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=32,
                        kernel_size=(41, 11),
                        stride=(2, 2),
                        padding=(20, 5),
                    ),
                    nn.BatchNorm1d(32),
                    nn.Hardtanh(0, 20, inplace=True),
                )
                self.out_features = (n_features + 20 * 2 - 41) // 2 + 1

            elif n_conv_layers == 2:
                self.conv_block = nn.Sequential(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=32,
                        kernel_size=(41, 11),
                        stride=(2, 2),
                        padding=(20, 5),
                    ),
                    nn.BatchNorm1d(32),
                    nn.Hardtanh(0, 20, inplace=True),
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=32,
                        kernel_size=(21, 11),
                        stride=(2, 1),
                        padding=(10, 5),
                    ),
                    nn.BatchNorm1d(32),
                    nn.Hardtanh(0, 20, inplace=True),
                )
                self.out_features = (n_features + 20 * 2 - 41) // 2 + 1
                self.out_features = (self.out_features + 10 * 2 - 21) // 2 + 1

            else:
                self.conv_block = nn.Sequential(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=32,
                        kernel_size=(41, 11),
                        stride=(2, 2),
                        padding=(20, 5),
                    ),
                    nn.BatchNorm1d(32),
                    nn.Hardtanh(0, 20, inplace=True),
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=32,
                        kernel_size=(21, 11),
                        stride=(2, 1),
                        padding=(10, 5),
                    ),
                    nn.BatchNorm1d(32),
                    nn.Hardtanh(0, 20, inplace=True),
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=96,
                        kernel_size=(21, 11),
                        stride=(2, 1),
                        padding=(10, 5),
                    ),
                    nn.BatchNorm1d(96),
                    nn.Hardtanh(0, 20, inplace=True),
                )
                self.out_features = (n_features + 10 * 2 - 21) // 2 + 1
                self.out_features = (self.out_features + 10 * 2 - 21) // 2 + 1
                self.out_features = (self.out_features + 20 * 2 - 41) // 2 + 1

    def forward(self, x, x_length):
        x = self.conv_block(x)
        return x, x_length // self.scaling


class DeepSpeech2(nn.Module):
    """
    Deep Speech 2 from http://proceedings.mlr.press/v48/amodei16.pdf
    """

    def __init__(
        self,
        n_features: int,
        n_tokens: int,
        conv_dim: int = 2,
        n_conv_layers: int = 2,
        n_rnn_layers: int = 5,
        fc_hidden: int = 512,
        **batch,
    ):
        super().__init__()

        self.conv_block = ConvBlock(
            conv_dim=conv_dim, n_conv_layers=n_conv_layers, n_features=n_features
        )

        rnn_input_size = self.conv_block.out_features
        rnn_output_size = fc_hidden * 2  # bidirectional

        self.gru_layers = [
            GRUBlock(input_size=rnn_input_size, hidden_size=rnn_output_size)
        ]

        for _ in range(n_rnn_layers - 1):
            self.gru_layers.append(
                GRUBlock(input_size=rnn_output_size, hidden_size=rnn_output_size)
            )

        self.fc = Sequential(
            nn.LayerNorm(rnn_output_size),
            nn.Linear(rnn_output_size, n_tokens, bias=False),
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

        log_probs = nn.functional.log_softmax(self.fc(outputs.transpose(0, 1)), dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

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
