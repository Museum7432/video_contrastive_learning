import lightning as L
from torch.utils.data import DataLoader, IterableDataset, Dataset
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch
from itertools import chain


class Vdataset(Dataset):
    def __init__(
        self,
        data_dir,
        num_positive_ins=2,
        num_negative_ins=8,
        add_noises=False,
        debugging=False,
    ):
        self.debugging = debugging
        self.num_positive_ins = num_positive_ins
        self.num_negative_ins = num_negative_ins
        self.add_noises = add_noises

        self.data_dir = data_dir

        self.v_indices = np.load(os.path.join(data_dir, "v_indices.npy"))

        self.v_features = np.float32(np.load(os.path.join(data_dir, "v_features.npy")))

        if debugging:
            d_len = len(self.v_indices) // 10
            self.v_indices = self.v_indices[:d_len]

    def __len__(self):
        return len(self.v_indices)

    def __getitem__(self, idx):

        _start, _end = self.v_indices[idx]

        if self.add_noises:
            max_cut = (_end - _start) // 2

            s = np.random.randint(max_cut + 1)
            e = np.random.randint(max_cut - s + 1)

            _start += s
            _end -= e

        frames_rep = self.v_features[_start:_end]
        seq_len = len(frames_rep)

        # craft nagetive queries
        pools = list(chain(range(0, _start), range(_end, len(self.v_features))))

        selected_frames_ids = np.random.choice(pools, size=seq_len * 2, replace=False)

        negative_queries = self.v_features[selected_frames_ids]
        if self.add_noises:
            noises = np.random.normal(loc=1, scale=0.05, size=selected_frames.shape).astype(
                "float32"
            )
            negative_queries = negative_queries * noises

        # craft positive queries from current video frames' representation
        selected_indices = np.random.randint(seq_len, size=seq_len) % (
            np.arange(seq_len) + 1
        )
        positive_queries = frames_rep[selected_indices]
        if self.add_noises:
            noises = np.random.normal(loc=1, scale=0.05, size=selected_frames.shape).astype(
                "float32"
            )
            positive_queries = positive_queries * noises

        # tensorized input
        frames_rep = torch.tensor(frames_rep)
        positive_queries = torch.tensor(positive_queries)
        negative_queries = torch.tensor(negative_queries)

        return {
            "frames_rep": frames_rep,
            "seq_lens": len(frames_rep),
            "positive_queries": positive_queries,
            "negative_queries": negative_queries,
        }


def collate_fn(items):

    frames_rep = pad_sequence([i["frames_rep"] for i in items], batch_first=True)

    seq_lens = torch.tensor([i["seq_lens"] for i in items])

    # positive_queries = pad_sequence(
    #     [i["positive_queries"] for i in items], batch_first=True
    # )
    # negative_queries = pad_sequence(
    #     [i["negative_queries"] for i in items], batch_first=True
    # )

    return {
        "frames_rep": frames_rep,
        "seq_lens": seq_lens,
        "positive_queries": [i["positive_queries"] for i in items],
        "negative_queries": [i["negative_queries"] for i in items],
    }


class Vdatamodule(L.LightningDataModule):
    def __init__(
        self,
        debugging=False,
        train_batch_size=4,
        valid_batch_size=4,
        num_workers=4,
        **other_args
    ):
        super().__init__()

        self.debugging = debugging

        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):

        self.train_dataset = Vdataset(data_dir="./data/train", debugging=self.debugging)

        self.val_dataset = Vdataset(data_dir="./data/val", debugging=self.debugging)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.valid_batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=False,
            num_workers=self.num_workers,
        )
