import gzip
import os
import shutil

import wget

from src.utils.io_utils import ROOT_PATH

URL_LINKS = {
    "bpe": "",
    "libri": "https://openslr.elda.org/resources/11/librispeech-vocab.txt",
    "lm": "https://www.openslr.org/resources/11/3-gram.pruned.3e-7.arpa.gz",
}


def download_vocab(vocab_type):
    data_dir = ROOT_PATH / "data" / "libri_lm"
    data_dir.mkdir(exist_ok=True, parents=True)
    path = str(data_dir) + f"/{vocab_type}_vocab.txt"
    print("Downloading vocab...")
    wget.download(URL_LINKS[vocab_type], path)
    print("\nVocab downloaded!")
    return path


def download_lm():
    data_dir = ROOT_PATH / "data/libri_lm"
    gz_path = str(data_dir) + "/uppercase_3e-7.arpa.gz"
    path = str(data_dir) + "/uppercase_3e-7.arpa"
    right_path = str(data_dir) + "/lowercase_3e-7.arpa"

    if os.path.exists(right_path):
        return right_path
    data_dir.mkdir(exist_ok=True, parents=True)

    print("Downloading language model...")
    wget.download(URL_LINKS["lm"], gz_path)
    with gzip.open(gz_path, "rb") as f_in:
        with open(path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    with open(path, "r") as f_upper:
        with open(right_path, "w") as f_lower:
            for line in f_upper:
                f_lower.write(line.lower())

    print("\nLanguage model downloaded!")
    return right_path
