import os
import torch
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizer

from dataset import CollectionDataset, QueryDataset


class V1IvfDrTrainSet(Dataset):
    def __init__(self, tripplets_and_qrels_dir, collection_memmap_dir,
        fixed_query_embeddings_dir, max_doc_length, hidden_size):

        super(V1IvfDrTrainSet, self).__init__()
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.max_doc_length = max_doc_length

        self.collection = CollectionDataset(collection_memmap_dir)

        self.qids, self.pids, self.labels = [], [], []
        official_train_dir = os.path.join(tripplets_and_qrels_dir, "tripplets.train.tsv")
        with open(official_train_dir) as fin:
            for line in tqdm(fin, desc="loading train tripples"):
                qid, pos_pid, neg_pid = line.split("\t")
                qid, pos_pid, neg_pid = int(qid), int(pos_pid), int(neg_pid)
                self.qids.append(qid)
                self.pids.append(pos_pid)
                self.labels.append(1)
                self.qids.append(qid)
                self.pids.append(neg_pid)
                self.labels.append(0)

        self.qrels = {}
        qrels_dir = os.path.join(tripplets_and_qrels_dir, "qrels.train.tsv")
        with open(qrels_dir) as fin:
            for line in tqdm(fin, desc="loading qrels"):
                qid, _, pid, _ = line.split("\t")
                qid, pid = int(qid), int(pid)
                if qid not in self.qrels:
                    self.qrels[qid] = []
                if pid not in self.qrels[qid]:
                    self.qrels[qid].append(pid)

        qid_cid_dir = os.path.join(fixed_query_embeddings_dir, "qid_cid.train.json")
        with open(qid_cid_dir) as fin:
            self.qid_cid = json.loads(fin.read())
        query_size = len(self.qid_cid)

        query_embeddings_memmap = np.memmap(
            os.path.join(fixed_query_embeddings_dir, "query_embeddings.train.memmap"),
            mode="c",
            shape=(query_size, hidden_size),
            dtype="float32"
        )
        self.query_embeddings = torch.from_numpy(query_embeddings_memmap)

        inverse_qid_dir = os.path.join(fixed_query_embeddings_dir, "inverse_qids.train.json")
        with open(inverse_qid_dir) as fin:
            self.inverse_qid = json.loads(fin.read())

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, idx):
        qid, pid = self.qids[idx], self.pids[idx]
        query_embeddings = self.query_embeddings[self.inverse_qid[str(qid)]]
        doc_input_ids = [self.cls_id] + self.collection[pid][:self.max_doc_length - 2] + [self.sep_id]
        top_centroid_ids = self.qid_cid[str(qid)]
        data = {
            "qid": qid,
            "pid": pid,
            "doc_input_ids": doc_input_ids,
            "query_embeddings": query_embeddings,
            "top_centroid_ids": top_centroid_ids,
            "rel_docs": self.qrels[qid]
        }
        return data

    @classmethod
    def _pack_tensor_2D(cls, lst, default, length=None):
        batch_size = len(lst)
        length = max(len(l) for l in lst) if length is None else length
        packed_tensor = default * torch.ones((batch_size, length), dtype=torch.int64)
        for i, l in enumerate(lst):
            packed_tensor[i,:len(l)] = torch.tensor(l, dtype=torch.int64)
        return packed_tensor
    
    @classmethod
    def collate_func(cls, batch):
        doc_input_ids = [x["doc_input_ids"] for x in batch]
        doc_attention_mask = [[1 for i in range(len(x))] for x in doc_input_ids]
        qids = [x["qid"] for x in batch]
        pids = [x["pid"] for x in batch]
        labels = [[i for i in range(len(pids)) if pids[i] in x["rel_docs"]] for x in batch]
        top_centroid_ids = [x["top_centroid_ids"] for x in batch]
        query_embeddings = torch.stack([x["query_embeddings"] for x in batch])
        data = {
            "doc_input_ids": cls._pack_tensor_2D(doc_input_ids, default=0),
            "doc_attention_mask": cls._pack_tensor_2D(doc_attention_mask, default=0),
            "labels": cls._pack_tensor_2D(labels, default=-1, length=len(batch)),
            "query_embeddings": query_embeddings,
            "top_centroid_ids": cls._pack_tensor_2D(top_centroid_ids, default=0)
        }
        return data