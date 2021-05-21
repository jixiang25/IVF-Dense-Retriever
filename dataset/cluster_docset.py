import os
import torch
import numpy as np

from torch.utils.data import Dataset


class ClusterDocset(Dataset):
    def __init__(self, doc_embedding_dir, hidden_size):
        super(ClusterDocset, self).__init__()
        self.hidden_size = hidden_size
        self.pids_memmap = np.memmap(
            os.path.join(doc_embedding_dir, "pids.memmap"),
            mode="c",
            dtype="int32"
        )
        self.collection_size = len(self.pids_memmap)
        self.doc_embedding_memmap = np.memmap(
            os.path.join(doc_embedding_dir, "doc_embeddings.memmap"),
            mode="c",
            shape=(self.collection_size, self.hidden_size),
            dtype="float32"
        )

    def __len__(self):
        return self.collection_size

    def __getitem__(self, idx):
        data = {
            "doc_embeddings": self.doc_embedding_memmap[idx],
            "pids": int(self.pids_memmap[idx])
        }
        return data

    @classmethod
    def collate_func(cls, batch):
        batch_size = len(batch)
        hidden_size = batch[0]["doc_embeddings"].shape[0]
        doc_embeddings = torch.zeros((batch_size, hidden_size))
        pids = list()
        for idx, x in enumerate(batch):
            doc_embeddings[idx] = torch.from_numpy(x["doc_embeddings"])
            pids.append(x["pids"])
        data = {
            "doc_embeddings": doc_embeddings,
            "pids": pids
        }
        return data