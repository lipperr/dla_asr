import torchaudio
from torch import Tensor, nn


class PitchShift(nn.Module):
    def __init__(self, sample_rate, n_steps, *args, **kwargs):
        super().__init__()
        self._aug = torchaudio.transforms.PitchShift(
            sample_rate, n_steps, *args, **kwargs
        )

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
