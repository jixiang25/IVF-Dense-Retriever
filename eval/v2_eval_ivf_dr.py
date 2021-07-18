import argparse
import os
import json
import logging
import torch
import numpy as np

from queue import PriorityQueue
from tqdm import tqdm
from transformers import BertConfig
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from dataset.dual_encoder_eval_docset import DualEncoderEvalDocSet
from utils.similarity_functions import dot_product, l1_distance, l2_distance
from cluster.cluster_with_certain_centroids import get_centroid_embeddings_from_cluster_result
from model.dual_encoder import DualEncoder
from model.v2_ivf_dense_retriever import IvfDenseRetriever
from eval import output_retrieved_ranking_result


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
    datefmt="%d %H:%M:%S",
    level=logging.INFO
)


def get_devices(device_use):
    if "cpu" in device_use or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        _, ids = device_use.split(":")
        device_id_list = [int(idx) for idx in ids.split(",")]
        device = torch.device("cuda:{}".format(device_id_list[0]))
        logger.warning("Only single gpu is supported, using {} currently".format(device))
    return device


def initilaze_embedding_memmap(embedding_dir, collection_size, embedding_size):
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)
    pids_memmap_dir = os.path.join(embedding_dir, "pids.memmap")
    pids_memmap = np.memmap(
        pids_memmap_dir,
        mode="w+",
        dtype="int32",
        shape=(collection_size, )
    )
    embedding_memmap_dir = os.path.join(embedding_dir, "doc_embeddings.memmap")
    embedding_memmap = np.memmap(
        embedding_memmap_dir,
        mode="w+",
        dtype="float32",
        shape=(collection_size, embedding_size)
    )
    return pids_memmap, embedding_memmap


def load_doc_embeddings(embedding_dir, hidden_size):
    pids_memmap_dir = os.path.join(embedding_dir, "pids.memmap")
    pids_memmap = np.memmap(
        pids_memmap_dir,
        mode="c",
        dtype="int32"
    )
    doc_embedding_memmap_dir = os.path.join(embedding_dir, "doc_embeddings.memmap")
    collection_size = len(pids_memmap)
    doc_embedding_memmap = np.memmap(
        doc_embedding_memmap_dir, 
        mode="c",
        shape=(collection_size, hidden_size),
        dtype="float32"
    )
    with open(os.path.join(embedding_dir, "pid_to_posid.json")) as fin:
        pid_to_posid = json.loads(fin.read())
    return pids_memmap, doc_embedding_memmap, pid_to_posid


def precompute_embeddings(args):
    # annotate devices
    device = get_devices(args.device)
    logger.info("   precompute on device:{}".format(device))

    # init model
    config = BertConfig.from_pretrained(args.checkpoint_dir)
    if args.model_type == "base-model":
        model = DualEncoder.from_pretrained(args.checkpoint_dir, config=config)
    elif args.model_type == "joint-model":
        # parameters for modified centroid embeddings do not need to load into IvfDenseRetriever
        model = IvfDenseRetriever.from_pretrained(args.checkpoint_dir, config=config)
    model.to(device)

    # loading eval collection dataset
    eval_doc_dataset = DualEncoderEvalDocSet(
        collection_memmap_dir=args.collection_memmap_dir,
        max_doc_length=args.max_doc_length
    )
    eval_doc_sampler = SequentialSampler(eval_doc_dataset)
    eval_doc_dataloader = DataLoader(
        eval_doc_dataset,
        batch_size=args.docs_batch_size,
        sampler=eval_doc_sampler,
        collate_fn=DualEncoderEvalDocSet.collate_func
    )
    collection_size = len(eval_doc_dataset)
    total_steps = len(eval_doc_dataloader)
    docid_to_memmapid = eval_doc_dataset.get_doc_id_to_memmap_id()

    # annotate output memmap
    doc_id_memmap, embedding_memmap = initilaze_embedding_memmap(
        embedding_dir=args.doc_embedding_dir,
        collection_size=collection_size,
        embedding_size=config.hidden_size
    )

    # generate doc embeddings
    model.eval()
    for batch in tqdm(eval_doc_dataloader, desc="precompute doc embedding", total=total_steps):
        doc_input_ids = batch["doc_input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pids = batch["pids"]
        doc_embedding = model(
            input_ids=doc_input_ids,
            attention_mask=attention_mask
        )
        doc_embedding = doc_embedding.detach().cpu().numpy()
        for idx, doc_id in enumerate(pids):
            memmap_id = docid_to_memmapid[doc_id]
            doc_id_memmap[memmap_id] = doc_id
            embedding_memmap[memmap_id] = doc_embedding[idx]


class EmbeddingEvalQuerySet(Dataset):
    def __init__(self, fixed_query_embeddings_dir, hidden_size):
        super(EmbeddingEvalQuerySet, self).__init__()
        self.qids = np.memmap(
            os.path.join(fixed_query_embeddings_dir, "qids.dev.memmap"),
            mode="c",
            dtype="int32"
        )
        self.query_size = len(self.qids)
        query_embeddings_memmap = np.memmap(
            os.path.join(fixed_query_embeddings_dir, "query_embeddings.dev.memmap"),
            mode="c",
            shape=(self.query_size, hidden_size),
            dtype="float32"
        )
        self.query_embeddings = torch.from_numpy(query_embeddings_memmap)

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


def eval_ivf_dense_retriever(args):
    # init devices
    device = get_devices(args.device)

    # load doc_embedding_memmap, pids_memmap, pid_to_posid
    pids_memmap, doc_embedding_memmap, pid_to_posid = load_doc_embeddings(args.doc_embedding_dir, args.hidden_size)

    # load centroid embeddings
    centroid_embeddings = get_centroid_embeddings_from_cluster_result(
        args.centroid_embedding_dir,
        args.cluster_num,
        args.hidden_size
    )
    centroid_embeddings = torch.from_numpy(centroid_embeddings).to(device)

    # load center_pids
    center_pids_dir = os.path.join(args.centroid_embedding_dir, "center_pids.json")
    with open(center_pids_dir) as fin:
        center_pids = json.loads(fin.read())
    
    # load eval queries dataset
    embedding_query_dataset = EmbeddingEvalQuerySet(
        fixed_query_embeddings_dir=args.fixed_query_embeddings_dir,
        hidden_size=args.hidden_size
    )
    embedding_query_sampler = SequentialSampler(embedding_query_dataset)
    embedding_query_dataloader = DataLoader(
        embedding_query_dataset,
        sampler=embedding_query_sampler,
        batch_size=args.queries_batch_size,
        collate_fn=EmbeddingEvalQuerySet.collate_func
    )
    total_steps = len(embedding_query_dataloader)    

    query_rank = {}
    # for batch in tqdm(eval_query_dataloader, desc="online inference with IVF", total=total_steps):
    for batch in embedding_query_dataloader:
        qids = batch["qids"]
        query_embedding = batch["query_embeddings"].to(device)

        score_to_centroids = l2_distance(query_embedding, centroid_embeddings)
        _, centroid_idxs = score_to_centroids.topk(k=args.nearest_centroid_num)
        centroid_idxs = centroid_idxs.cpu().detach().numpy()

        for idx, qid in tqdm(enumerate(qids)):
            query = query_embedding[idx].cpu().detach().numpy()
            if qid not in query_rank:
                query_rank[qid] = PriorityQueue(maxsize=args.hit_num)
            cur_pq = query_rank[qid]
            for i in range(args.nearest_centroid_num):
                assigned_centroid = str(centroid_idxs[idx][i])
                for pid in center_pids[assigned_centroid]:
                    doc_pos = int(pid_to_posid[str(pid)])
                    doc = doc_embedding_memmap[doc_pos]
                    score = float(-np.sum((query - doc) ** 2))
                    if cur_pq.full():
                        lowest_score, lowest_doc_id = cur_pq.get_nowait()
                        if lowest_score >= score:
                            cur_pq.put_nowait((lowest_score, lowest_doc_id))
                        else:
                            cur_pq.put_nowait((score, pid))
                    else:
                        cur_pq.put_nowait((score, pid))
    
    # output result for metric calculating
    output_retrieved_ranking_result(
        retrieved_result_dir=args.retrieved_result_dir,
        retrieved_result=query_rank,
        hit_num=args.hit_num
    )


def eval_query_hit_centroid_rate(args):
    # init devices
    device = get_devices(args.device)

    # load doc_embedding_memmap, pids_memmap, pid_to_posid
    pids_memmap, doc_embedding_memmap, pid_to_posid = load_doc_embeddings(args.doc_embedding_dir, args.hidden_size)

    # load centroid embeddings
    centroid_embeddings = get_centroid_embeddings_from_cluster_result(
        args.centroid_embedding_dir,
        args.cluster_num,
        args.hidden_size
    )
    centroid_embeddings = torch.from_numpy(centroid_embeddings).to(device)

    # load center_pids
    center_pids_dir = os.path.join(args.centroid_embedding_dir, "center_pids.json")
    with open(center_pids_dir) as fin:
        center_pids = json.loads(fin.read())
    
    # load eval queries dataset
    embedding_query_dataset = EmbeddingEvalQuerySet(
        fixed_query_embeddings_dir=args.fixed_query_embeddings_dir,
        hidden_size=args.hidden_size
    )
    embedding_query_sampler = SequentialSampler(embedding_query_dataset)
    embedding_query_dataloader = DataLoader(
        embedding_query_dataset,
        sampler=embedding_query_sampler,
        batch_size=args.queries_batch_size,
        collate_fn=EmbeddingEvalQuerySet.collate_func
    )
    total_steps = len(embedding_query_dataloader)    

    dev_qrels_dir = os.path.join(args.tripplets_and_qrels_dir, "qrels.dev.small.tsv")
    dev_qrels = dict()
    with open(dev_qrels_dir) as fin:
        for line in fin:
            qid, _, pid, _ = line.split("\t")
            qid, pid = int(qid), int(pid)
            if qid not in dev_qrels:
                dev_qrels[qid] = []
            dev_qrels[qid].append(pid)

    ans_q, ans_p = 0, 0
    # for batch in tqdm(eval_query_dataloader, desc="online inference with IVF", total=total_steps):
    for batch in embedding_query_dataloader:
        qids = batch["qids"]
        query_embedding = batch["query_embeddings"].to(device)

        score_to_centroids = l2_distance(query_embedding, centroid_embeddings)
        _, centroid_idxs = score_to_centroids.topk(k=args.nearest_centroid_num)
        centroid_idxs = centroid_idxs.cpu().detach().numpy()

        for idx, qid in tqdm(enumerate(qids)):
            rel_doc_lst = dev_qrels[qid]
            for rel_doc in rel_doc_lst:
                ans_p += 1
                for i in range(args.nearest_centroid_num):
                    assigned_centroid = str(centroid_idxs[idx][i])
                    if rel_doc in center_pids[assigned_centroid]:
                        ans_q += 1
                        break
    
    print("{} / {} = {}".format(ans_q, ans_p, ans_q / ans_p))


def eval_ivf_dense_retriever(args):
    # init devices
    device = get_devices(args.device)

    # load doc_embedding_memmap, pids_memmap, pid_to_posid
    pids_memmap, doc_embedding_memmap, pid_to_posid = load_doc_embeddings(args.doc_embedding_dir, args.hidden_size)

    # load centroid embeddings
    centroid_embeddings = get_centroid_embeddings_from_cluster_result(
        args.centroid_embedding_dir,
        args.cluster_num,
        args.hidden_size
    )
    centroid_embeddings = torch.from_numpy(centroid_embeddings).to(device)

    # load center_pids
    center_pids_dir = os.path.join(args.centroid_embedding_dir, "center_pids.json")
    with open(center_pids_dir) as fin:
        center_pids = json.loads(fin.read())
    
    # load eval queries dataset
    embedding_query_dataset = EmbeddingEvalQuerySet(
        fixed_query_embeddings_dir=args.fixed_query_embeddings_dir,
        hidden_size=args.hidden_size
    )
    embedding_query_sampler = SequentialSampler(embedding_query_dataset)
    embedding_query_dataloader = DataLoader(
        embedding_query_dataset,
        sampler=embedding_query_sampler,
        batch_size=args.queries_batch_size,
        collate_fn=EmbeddingEvalQuerySet.collate_func
    )
    total_steps = len(embedding_query_dataloader)    

    dev_qrels_dir = os.path.join(args.tripplets_and_qrels_dir, "qrels.dev.small.tsv")
    dev_qrels = dict()
    with open(dev_qrels_dir) as fin:
        for line in fin:
            qid, _, pid, _ = line.split("\t")
            qid, pid = int(qid), int(pid)
            if qid not in dev_qrels:
                dev_qrels[qid] = []
            dev_qrels[qid].append(pid)

    ans_q, ans_p = 0, 0
    # for batch in tqdm(eval_query_dataloader, desc="online inference with IVF", total=total_steps):
    for batch in embedding_query_dataloader:
        qids = batch["qids"]
        query_embedding = batch["query_embeddings"].to(device)

        score_to_centroids = l2_distance(query_embedding, centroid_embeddings)
        _, centroid_idxs = score_to_centroids.topk(k=args.nearest_centroid_num)
        centroid_idxs = centroid_idxs.cpu().detach().numpy()

        for idx, qid in tqdm(enumerate(qids)):
            rel_doc_lst = dev_qrels[qid]
            for rel_doc in rel_doc_lst:
                ans_p += 1
                for i in range(args.nearest_centroid_num):
                    assigned_centroid = str(centroid_idxs[idx][i])
                    if rel_doc in center_pids[assigned_centroid]:
                        ans_q += 1
                        break
    
    print("{} / {} = {}".format(ans_q, ans_p, ans_q / ans_p))


def get_args():
    parser = argparse.ArgumentParser()
    # action related args
    parser.add_argument("--do_precompute", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_eval_query_hit", action="store_true")
    # path related args
    parser.add_argument("--collection_memmap_dir", type=str, default="./data/msmarco-passage/avg_L2_CE_lr1e-5/collection_memmap/complete")
    parser.add_argument("--doc_embedding_dir", type=str, default="./data/msmarco-passage/avg_L2_CE_lr1e-5/embeddings")
    parser.add_argument("--centroid_embedding_dir", type=str, default="./data/msmarco-passage/cls_L2_CE_lr1e-5/centroid_embeddings")
    parser.add_argument("--checkpoint_dir", type=str, default="./data/msmarco-passage/cls_L2_CE_lr1e-5/ivf_dr_checkpoint")
    parser.add_argument("--fixed_query_embeddings_dir", type=str, default="./data/msmarco-passage/cls_L2_CE_lr1e-5/fixed_query_embeddings")
    parser.add_argument("--tripplets_and_qrels_dir", type=str, default="./data/msmarco-passage/tripplets_and_qrels")
    parser.add_argument("--retrieved_result_dir", type=str, default="./data/msmarco-passage/cls_L2_CE_lr1e-5/retrieved_result")
    # model related args
    parser.add_argument("--model_type", type=str, default="base-model")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--docs_batch_size", type=int, default=40)
    parser.add_argument("--queries_batch_size", type=int, default=500)
    parser.add_argument("--max_doc_length", type=int, default=256)
    parser.add_argument("--max_query_length", type=int, default=32)
    # cluster related args
    parser.add_argument("--cluster_num", type=int, default=1000)
    parser.add_argument("--nearest_centroid_num", type=int, default=5)
    parser.add_argument("--hit_num", type=int, default=1000)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.do_precompute:
        precompute_embeddings(args)
    if args.do_eval:
        eval_ivf_dense_retriever(args)
    if args.do_eval_query_hit:
        eval_query_hit_centroid_rate(args)


if __name__ == "__main__":
    main()