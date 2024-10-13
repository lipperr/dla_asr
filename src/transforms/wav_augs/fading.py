import torchaudio
from torch import Tensor, nn


class Fading(nn.Module):
    def __init__(self, alpha=0.5, *args, **kwargs):
        super().__init__()
        self.alpha = alpha

    def __call__(self, data: Tensor):
        fade_len = int(data.shape[-1] * self.alpha)
        self._aug = torchaudio.transforms.Fade(fade_len, fade_len)
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
