import torchaudio
from torch import Tensor, nn


class Fading(nn.Module):
    def __init__(self, fade_in_len, fade_out_len, *args, **kwargs):
        super().__init__()
        self._aug = torchaudio.transforms.Fade(
            fade_in_len, fade_out_len, *args, **kwargs
        )

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
