import argparse
import os
import json
import logging
import torch
import numpy as np

from queue import PriorityQueue
from tqdm import tqdm
from transformers import BertConfig
from torch.utils.data import DataLoader, SequentialSampler
from dataset.cluster_docset import ClusterDocset
from utils.similarity_functions import l2_distance


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


def get_centroid_embeddings_from_cluster_result(centroid_embedding_dir, cluster_num, hidden_size):
    centroid_memmap = np.memmap(
        os.path.join(centroid_embedding_dir, "centroid_embeddings.memmap"),
        mode="c",
        shape=(cluster_num, hidden_size),
        dtype="float32"
    )
    return centroid_memmap


def dump_centroid_embeddings_from_model_parameters(checkpoint_dir, centroid_embedding_dir, cluster_num, hidden_size):
    if not os.path.exists(centroid_embedding_dir):
        os.makedirs(centroid_embedding_dir)
    ckpt_model_dir = os.path.join(checkpoint_dir, "pytorch_model.bin")
    checkpoint = torch.load(ckpt_model_dir, map_location="cpu")
    centroid_embeddings = checkpoint["centroid_embeddings"].detach().numpy()
    centroid_memmap = np.memmap(
        os.path.join(centroid_embedding_dir, "centroid_embeddings.memmap"),
        mode="w+",
        shape=(cluster_num, hidden_size),
        dtype="float32"
    )
    for i in range(cluster_num):
        centroid_memmap[i] = centroid_embeddings[i]


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


def cluster_with_certain_centorids(args):
    # annotate device
    device = get_devices(device_use=args.device)
    # load centroid embeddings
    if args.centroid_source == "cluster-result":
        centroid_embeddings = get_centroid_embeddings_from_cluster_result(
            centroid_embedding_dir=args.centroid_embedding_dir,
            cluster_num=args.cluster_num,
            hidden_size=args.hidden_size
        )
    elif args.centroid_source == "model-parameters":
        dump_centroid_embeddings_from_model_parameters(
            checkpoint_dir=args.checkpoint_dir,
            centroid_embedding_dir=args.centroid_embedding_dir,
            cluster_num=args.cluster_num,
            hidden_size=args.hidden_size
        )
        centroid_embeddings = get_centroid_embeddings_from_cluster_result(
            centroid_embedding_dir=args.centroid_embedding_dir,
            cluster_num=args.cluster_num,
            hidden_size=args.hidden_size
        )
    else:
        raise ValueError("`{}` is not supported for argument `centroid_source`!".format(args.centroid_source))
    centroid_embeddings = torch.from_numpy(centroid_embeddings).to(device)
    # load document cluster embeddings
    cluster_docset = ClusterDocset(args.doc_embedding_dir, args.hidden_size)
    cluster_sampler = SequentialSampler(cluster_docset)
    cluster_dataloader = DataLoader(
        dataset=cluster_docset,
        sampler=cluster_sampler,
        batch_size=args.cluster_batch_size,
        collate_fn=ClusterDocset.collate_func
    )
    # assign each document to a cluster centroid
    center_pids = {centroid: [] for centroid in range(args.cluster_num)}
    total_steps = len(cluster_dataloader)
    for batch in tqdm(cluster_dataloader, desc="clustring with certain centroids", total=total_steps):
        doc_embeddings = batch["doc_embeddings"].to(device)
        pids = batch["pids"]
        score_matrix = l2_distance(doc_embeddings, centroid_embeddings)
        _, centroid_ids = score_matrix.topk(k=1)
        for idx, pid in enumerate(pids):
            centroid_id = int(centroid_ids[idx][0].item())
            center_pids[centroid_id].append(pid)
    center_pids_dir = os.path.join(args.centroid_embedding_dir, "center_pids.json")
    with open(center_pids_dir, "w") as fout:
        fout.write(json.dumps(center_pids))


def get_args():
    parser = argparse.ArgumentParser()
    # data dir related args
    parser.add_argument("--doc_embedding_dir", type=str, default="./data/msmarco-passage/avg_L2_CE_lr1e-5/embeddings")
    parser.add_argument("--centroid_embedding_dir", type=str, default="./data/msmarco-passage/cls_L2_CE_lr1e-5/centroid_embeddings_before")
    parser.add_argument("--checkpoint_dir", type=str, default="./data/msmarco-passage/cls_L2_CE_lr1e-5/data/ivf_dr_checkpoint")
    # cluster info related args
    parser.add_argument("--centroid_source", type=str, help="`model-parameters` or `cluster-result`")
    parser.add_argument("--cluster_num", type=int, default=1000)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cluster_batch_size", type=int, default=1000)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    cluster_with_certain_centorids(args)


if __name__ == "__main__":
    main()