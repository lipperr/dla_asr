import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    batch = dict()
    keys = dataset_items[0].keys()

    for key in keys:
        if key == "spectrogram" or key == "text_encoded":
            batch[f"{key}_length"] = torch.tensor(
                data=[item[key].shape[-1] for item in dataset_items]
            )
            batch[key] = pad_sequence(
                [item[key].squeeze(dim=0).t() for item in dataset_items],
                batch_first=True,
            )
        else:
            batch[key] = [item[key] for item in dataset_items]
    batch["spectrogram"] = batch["spectrogram"].permute(0, 2, 1)
    print("RETURNED BATCH")
    return batch
