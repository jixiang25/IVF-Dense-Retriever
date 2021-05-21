import argparse
import os
import json
import logging
import faiss
import numpy as np

from sklearn.cluster import KMeans


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
    datefmt="%d %H:%M:%S",
    level=logging.INFO
)


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
    # create a dictionary to map pid to location index in doc_embedding matrix
    pid_to_posid = {int(pid):idx for idx, pid in enumerate(pids_memmap)}
    with open(os.path.join(embedding_dir, "pid_to_posid.json"), "w") as fout:
        fout.write(json.dumps(pid_to_posid))
    return pids_memmap, doc_embedding_memmap


def k_means_clustering(args):
    pids_memmap, doc_embedding_memmap = load_doc_embeddings(args.doc_embedding_dir, args.hidden_size)
    
    quantizer = faiss.IndexFlatL2(args.hidden_size)
    index = faiss.IndexIVFFlat(quantizer, args.hidden_size, args.cluster_num, faiss.METRIC_L2)
    index.train(doc_embedding_memmap)
    index.add(doc_embedding_memmap)
    centroids = quantizer.reconstruct_n(0, args.cluster_num)

    if not os.path.exists(args.centroid_embedding_dir):
        os.makedirs(args.centroid_embedding_dir)
    centroid_memmap = np.memmap(
        os.path.join(args.centroid_embedding_dir, "centroids_embeddings.memmap"),
        mode="w+",
        shape=(args.cluster_num, args.hidden_size),
        dtype="float32",
    )
    for idx in range(args.cluster_num):
        centroid_memmap[idx] = centroids[idx]


# def k_means_clustering(args):
#     pids_memmap, doc_embedding_memmap = load_doc_embeddings(args.doc_embedding_dir, args.hidden_size)
#     kmeans = faiss.Kmeans(args.hidden_size, args.cluster_num, niter=args.train_epoch, verbose=True)
#     kmeans.train(doc_embedding_memmap)
#     if not os.path.exists(args.centroid_embedding_dir):
#         os.makedirs(args.centroid_embedding_dir)
#     centroid_memmap = np.memmap(
#         os.path.join(args.centroid_embedding_dir, "centroids_before_training.memmap"),
#         mode="w+",
#         shape=(args.cluster_num, args.hidden_size),
#         dtype="float32",
#     )
#     for idx in range(args.cluster_num):
#         centroid_memmap[idx] = kmeans.centroids[idx]


# def k_means_clustering(args):
#     pids_memmap, doc_embedding_memmap = load_doc_embeddings(args.doc_embedding_dir, args.hidden_size)
#     kmeans = KMeans(n_clusters=args.cluster_num, verbose=True)
#     kmeans.fit(doc_embedding_memmap)
#     logger.info(kmeans.score(doc_embedding_memmap))
#     if not os.path.exists(args.centroid_embedding_dir):
#         os.makedirs(args.centroid_embedding_dir)
#     centroid_memmap = np.memmap(
#         os.path.join(args.centroid_embedding_dir, "centroids_before_training.memmap"),
#         mode="w+",
#         shape=(args.cluster_num, args.hidden_size),
#         dtype="float32",
#     )
#     for idx in range(args.cluster_num):
#         centroid_memmap[idx] = kmeans.cluster_centers_[idx]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_embedding_dir", type=str, default="./data/msmarco-passage/avg_L2_CE_lr1e-5/embeddings")
    parser.add_argument("--centroid_embedding_dir", type=str, default="./data/msmarco-passage/avg_L2_CE_lr1e-5/centroid_embeddings")
    parser.add_argument("--cluster_num", type=int, default=1000)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--train_epoch",type=int, default=10)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    k_means_clustering(args)


if __name__ == "__main__":
    main()
