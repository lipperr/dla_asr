# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains an example solution for ASR task with PyTorch. This template branch is a part of the [HSE DLA course](https://github.com/markovka17/dla) ASR homework.

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw1_asr).

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

## How to reproduce the best model
Run the following command:
```bash
python3 train.py -cn=train_argmax
```
The logs from training the model with config train_argmax.yaml are [here](https://drive.google.com/file/d/14ZhkszU6IMt0f0okZ6SvNknMYzlLHKcO/view?usp=sharing)

## Additional features

-  BeamSearch CTC decoder and a pretrained LM model
-  Custom vocabulary
-  Resume training from a pretrained model state

## Auxillary functions

- *wer_cer_compute.py* calculates and prints metrics given the path to ground truth and predicted transcriptions
- *download_model.py* downloads the best model to a path given as an argument

## How to run inference on the final model

Run the command
```bash
python3 inference.py -cn=inference
```
**inference.yaml** containes a parameter *from_pretrained*. If *from_pretrained* is not None, it's the path to a model on which the inference would run. By default, *from_pretrained* is the path to the best model.

Parameter *metrics* containes a config with metrics on which to run inference. Default is *bs_lm_metrics* (pretrained LM with BeamSearch).

If you want to run inference on a **custom dataset**, specify batch_size N and paths to audio and/or transcriptions as follows:

```bash
python3 inference.py -cn=inference datasets="custom" datasets.test.audio_dir="your path" datasets.test.transcription_dir="your path"  dataloader.batch_size=N
```

## WER and CER of the final model

test-clean CER_(Argmax): **0.07**
test-clean WER_(Argmax): **0.23**

LM with BeamSearch significantly improves metrics:

test-clean CER_(LM-BeamSearch): **0.06**
test-clean WER_(LM-BeamSearch): **0.16**

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
