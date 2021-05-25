import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, BertConfig

from dataset import CollectionDataset, QueryDataset
from model.dual_encoder import DualEncoder
from utils.similarity_functions import l2_distance


class IvfDrQueryDataset(Dataset):
    def __init__(self, mode, tokenize_dir, max_query_length):
        super(IvfDrQueryDataset, self).__init__()
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.max_query_length = max_query_length
        self.queries = QueryDataset(tokenize_dir, mode)
        self.qids = []
        for qid in self.queries.queries.keys():
            self.qids.append(qid)

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, idx):
        qid = self.qids[idx]
        query_input_ids = [self.cls_id] + self.queries[self.qids[idx]][:self.max_query_length - 2] + [self.sep_id]
        data = {
            "query_input_ids": query_input_ids,
            "qid": qid
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
        qids = [x["qid"] for x in batch]
        query_input_ids = [x["query_input_ids"] for x in batch]
        attention_mask = [[1 for i in range(len(x))] for x in query_input_ids]
        data = {
            "qids": qids,
            "query_input_ids": cls._pack_tensor_2D(query_input_ids, default=0),
            "attention_mask": cls._pack_tensor_2D(attention_mask, default=0)
        }
        return data


class EmbeddingQueryDataset(Dataset):
    def __init__(self, query_embedding_dir, mode, hidden_size, device):
        super(EmbeddingQueryDataset, self).__init__()
        self.qids = np.memmap(
            os.path.join(query_embedding_dir, "qids.{}.memmap".format(mode)),
            mode="c",
            dtype="int32"
        )
        self.query_size = len(self.qids)
        query_embeddings_memmap = np.memmap(
            os.path.join(query_embedding_dir, "query_embeddings.{}.memmap".format(mode)),
            mode="c",
            shape=(self.query_size, hidden_size),
            dtype="float32"
        )
        self.query_embeddings = torch.from_numpy(query_embeddings_memmap).to(device)
    
    def __len__(self):
        return self.query_size

    def __getitem__(self, idx):
        data = {
            "query_embeddings": self.query_embeddings[idx],
            "qids": int(self.qids[idx])
        }
        return data

    @classmethod
    def collate_func(cls, batch):
        qids = [x["qids"] for x in batch]
        query_embeddings = torch.stack([x["query_embeddings"] for x in batch])
        data = {
            "qids": qids,
            "query_embeddings": query_embeddings
        }
        return data


def get_device(args):
    if "cpu" in args.device or not torch.cuda.is_available():
        device = torch.device("cpu")
        args.gpu_count = 0
    else:
        _, ids = args.device.split(":")
        device_id_list = [int(idx) for idx in ids.split(",")]
        device = torch.device("cuda:{}".format(device_id_list[0]))
        args.gpu_count = len(device_id_list)
    return device


def inference_fixed_query_embeddings(args):
    device = get_device(args)
    
    query_dataset = IvfDrQueryDataset(
        mode=args.mode,
        tokenize_dir=args.tokenize_dir,
        max_query_length=args.max_query_length
    )
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(
        query_dataset,
        batch_size=args.batch_size,
        sampler=query_sampler,
        collate_fn=IvfDrQueryDataset.collate_func
    )

    config = BertConfig.from_pretrained(args.load_model_path)
    config.return_dict = False
    model = DualEncoder.from_pretrained(args.load_model_path, config=config)
    model.to(device)

    query_size = len(query_dataset)
    if not os.path.exists(args.fixed_query_embedding_dir):
        os.makedirs(args.fixed_query_embedding_dir)
    fixed_query_embeddings = np.memmap(
        os.path.join(args.fixed_query_embedding_dir, "query_embeddings.{}.memmap".format(args.mode)),
        mode="w+",
        shape=(query_size, args.hidden_size),
        dtype="float32"
    )
    qids_memmap = np.memmap(
        os.path.join(args.fixed_query_embedding_dir, "qids.{}.memmap".format(args.mode)),
        mode="w+",
        shape=(query_size,),
        dtype="int32"
    )
    inverse_qids_dict = dict()

    model.eval()
    pos_id = 0
    for batch in tqdm(query_dataloader):
        qids = batch["qids"]
        query_input_ids = batch["query_input_ids"].to(device)
        query_attention_mask = batch["attention_mask"].to(device)

        query_embeddings_tensor = model(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask
        )
        query_embeddings_ndarray = query_embeddings_tensor.cpu().detach().numpy()

        for idx, qid in enumerate(qids):
            qids_memmap[pos_id] = qid
            fixed_query_embeddings[pos_id] = query_embeddings_ndarray[idx]
            inverse_qids_dict[qid] = pos_id
            pos_id += 1
            
    with open(os.path.join(args.fixed_query_embedding_dir, "inverse_qids.{}.json".format(args.mode)), "w") as fout:
        fout.write(json.dumps(inverse_qids_dict))


def match_query_centroid(args):
    device = get_device(args)

    centroid_embeddings_memmap = np.memmap(
        os.path.join(args.centroid_embedding_dir, "centroid_embeddings.memmap"),
        mode="c",
        dtype="float32",
        shape=(args.cluster_num, args.hidden_size)
    )
    centroid_embeddings = torch.from_numpy(centroid_embeddings_memmap).to(device)

    embedding_query_dataset = EmbeddingQueryDataset(
        query_embedding_dir=args.fixed_query_embedding_dir,
        mode=args.mode,
        hidden_size=args.hidden_size,
        device=device
    )
    embedding_query_sampler = SequentialSampler(embedding_query_dataset)
    embedding_query_dataloader = DataLoader(
        embedding_query_dataset,
        batch_size=args.embedding_batch_size,
        sampler=embedding_query_sampler,
        collate_fn=EmbeddingQueryDataset.collate_func
    )

    query_centroid_dict = dict()
    for batch in tqdm(embedding_query_dataloader):
        qids = batch["qids"]
        query_embeddings = batch["query_embeddings"]
        # print(query_embeddings)
        # print(query_embeddings.size())
        distance_matrix = l2_distance(query_embeddings, centroid_embeddings)
        _, top_centroid_ids = distance_matrix.topk(k=args.hit_cluster_num, dim=-1)
        for idx, qid in enumerate(qids):
            if qid not in query_centroid_dict:
                query_centroid_dict[qid] = []
            for i in range(args.hit_cluster_num):
                centroid_id = int(top_centroid_ids[idx][i].item())
                query_centroid_dict[qid].append(centroid_id)
    
    with open(os.path.join(args.fixed_query_embedding_dir, "qid_cid.{}.json".format(args.mode)), "w") as fout:
        fout.write(json.dumps(query_centroid_dict))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_inference_query_embeddings", action="store_true")
    parser.add_argument("--do_match_query_centroid", action="store_true")

    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--tokenize_dir", type=str, default="./data/msmarco-passage/avg_L2_CE_lr1e-5/tokenize")
    parser.add_argument("--fixed_query_embedding_dir", type=str, default="./data/msmarco-passage/avg_L2_CE_lr1e-5/fixed_query_embeddings")
    parser.add_argument("--load_model_path", type=str, default="./data/msmarco-passage/avg_L2_CE_lr1e-5/checkpoints/step-270000")
    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--centroid_embedding_dir", type=str, default="./data/msmarco-passage/avg_L2_CE_lr1e-5/centroid_embeddings")
    parser.add_argument("--cluster_num", type=int, default=1000)
    parser.add_argument("--embedding_batch_size", type=int, default=100)
    parser.add_argument("--hit_cluster_num", type=int, default=5)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.do_inference_query_embeddings:
        inference_fixed_query_embeddings(args)
    if args.do_match_query_centroid:
        match_query_centroid(args)