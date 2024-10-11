import os
import re
from string import ascii_lowercase

import kenlm
import torch
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel

from ..utils.load_utils import download_vocab


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, vocab_type=None, lm_path=None, use_bpe=False, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if vocab_type is None:
            alphabet = list(ascii_lowercase + " ")
        else:
            vocab_path = download_vocab(vocab_type)
            assert os.path.exists(vocab_path), "Vocab path does not exist."
            with open(vocab_path) as f:
                alphabet = [t.lower() for t in f.read().strip().split("\n")]
            if " " not in alphabet:
                alphabet.append(" ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        if lm_path:
            assert os.path.exists(lm_path), "LM path does not exist."
            kenlm_model = kenlm.Model(lm_path)
            lm = LanguageModel(kenlm_model, alphabet)
            self.decoder_lm = BeamSearchDecoderCTC(Alphabet(self.vocab, False), lm)

        self.decoder = BeamSearchDecoderCTC(Alphabet(self.vocab, False), None)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        decoded = []
        empty_tok_ind = self.char2ind[self.EMPTY_TOK]
        last = empty_tok_ind
        for ind in inds:
            if last == ind:
                continue
            if ind != empty_tok_ind:
                decoded.append(self.ind2char[ind])
            last = ind
        return "".join(decoded)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text

    def ctc_beamsearch(
        self, probs: torch.Tensor, type="lm", beam_size=10
    ) -> list[dict[str, float]]:
        if type == "lm":
            return [
                {
                    "hypothesis": self.decoder_lm.decode(probs, beam_size),
                    "probability": 1.0,
                }
            ]

        elif type == "nolm":
            return [
                {
                    "hypothesis": self.decoder_no_lm.decode(probs, beam_size),
                    "probability": 1.0,
                }
            ]

        else:
            dp = {("", self.EMPTY_TOK): 1.0}
            for next_probs in probs:
                dp0 = {}
                for ind, next_prob in enumerate(next_probs):
                    curr = self.ind2char[ind]
                    for (pref, last), prob in dp.items():
                        if last != curr and curr != self.EMPTY_TOK:
                            new_pref = pref + curr
                        else:
                            new_pref = pref
                        dp0[(new_pref, curr)] = (
                            dp0.get((new_pref, curr), 0.0) + prob * next_prob
                        )
                dp = dp0.copy()

                dp = dict(sorted(list(dp.items()), key=lambda x: -x[1])[:beam_size])
            sorted_dp_list = sorted(dp.items(), key=lambda x: -x[1])

            dp = [
                {"hypothesis": prefix, "probability": prob.item()}
                for (prefix, _), prob in sorted_dp_list
            ]
            return dp
