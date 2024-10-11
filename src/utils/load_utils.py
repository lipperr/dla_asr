import os

import wget

from src.utils.io_utils import ROOT_PATH

URL_LINKS = {
    "bpe": "",
    "libri": "https://openslr.elda.org/resources/11/librispeech-vocab.txt",
}


def download_vocab(vocab_type):
    data_dir = ROOT_PATH / "data" / "libri_lm"
    data_dir.mkdir(exist_ok=True, parents=True)
    path = str(data_dir) + f"/{vocab_type}_vocab.txt"
    print("Downloading vocab...")
    wget.download(URL_LINKS[vocab_type], path)
    print("\nVocab downloaded!")
    return path
