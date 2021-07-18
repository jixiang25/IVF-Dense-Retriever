import argparse
import os
import logging
import torch
import faiss
import numpy as np
from tqdm import tqdm
from torch.utils.data import SequentialSampler, DataLoader
from transformers import BertConfig

from dataset.dual_encoder_eval_queryset import DualEncoderEvalQuerySet
from model.dual_encoder import DualEncoder


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
    datefmt="%d %H:%M:%S",
    level=logging.INFO
)


def load_doc_embeddings(embedding_dir, hidden_size):
    pids_memmap = np.memmap(
        os.path.join(embedding_dir, "pids.memmap"),
        mode="c",
        dtype="int32",
    )
    collection_size = len(pids_memmap)
    doc_embedding_memmap = np.memmap(
        os.path.join(embedding_dir, "doc_embeddings.memmap"),
        mode="c",
        shape=(collection_size, hidden_size),
        dtype="float32"
    )
    return doc_embedding_memmap, pids_memmap


def build_faiss_index(args):
    quantizer = faiss.IndexFlatL2(args.hidden_size)
    index = faiss.IndexIVFFlat(
        quantizer, 
        args.hidden_size,
        args.center_nums,
        faiss.METRIC_L2
    )
    index.nprobe=args.nprobe
    doc_embedding, pids = load_doc_embeddings(args.embedding_dir, args.hidden_size)
    logging.info("=======  Clustering document embeddings  =======")
    index.train(doc_embedding)
    logging.info("=======  Finish Clustering document embeddings  =======")
    index.add(doc_embedding)
    if not os.path.exists(args.index_dir):
        os.makedirs(args.index_dir)
    faiss.write_index(
        index,
        os.path.join(args.index_dir, "ivf-{}-{}.index".format(args.center_nums, args.nprobe))
    )


def retrieve_from_index(args):
    # loading model
    config = BertConfig.from_pretrained(args.load_model_path)
    model = DualEncoder.from_pretrained(args.load_model_path, config=config)
    model.to(args.device)

    # loading eval queries dataset
    eval_query_dataset = DualEncoderEvalQuerySet(
        tokenize_dir=args.tokenize_dir,
        max_query_length=args.max_query_length,
        mode="test"
    )
    eval_query_sampler = SequentialSampler(eval_query_dataset)
    eval_query_dataloader = DataLoader(
        eval_query_dataset,
        sampler=eval_query_sampler,
        batch_size=args.queries_batch_size,
        collate_fn=DualEncoderEvalQuerySet.collate_func
    )
    total_steps = len(eval_query_dataloader)

    # loading faiss index
    index = faiss.read_index(
        os.path.join(args.index_dir, "ivf-{}-{}.index".format(args.center_nums, args.nprobe))
    )

    # loading pos_id to pid map
    pids_memmap = np.memmap(
        os.path.join(args.embedding_dir, "pids.memmap"),
        mode="c",
        dtype="int32",
    )

    result_dict = {}
    for batch in tqdm(eval_query_dataloader, desc="online retrieve docs", total=total_steps):
        query_input_ids = batch["query_input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        qids = batch["qids"]
        query_embedding = model(
            input_ids=query_input_ids,
            attention_mask=attention_mask
        ).detach().cpu().numpy()
        scores, ids = index.search(query_embedding, args.hit_num)
        for idx, qid in enumerate(qids):
            result_dict[qid] = list((pid, s) for pid, s in zip(ids[idx], scores[idx]))
    
    if not os.path.exists(args.retrieved_result_dir):
        os.makedirs(args.retrieved_result_dir)
    with open(os.path.join(args.retrieved_result_dir, "top1000.tsv"), "w") as fout:
        for qid in result_dict:
            rank = 1
            for pid, score in result_dict[qid]:
                fout.write("{}\t{}\t{}\t{}\n".format(qid, pid, score, rank))
                rank += 1
                

def get_args():
    parser = argparse.ArgumentParser()
    # action args
    parser.add_argument("--do_build", action="store_true")
    parser.add_argument("--do_retrieve", action="store_true")
    # model args
    parser.add_argument("--load_model_path", type=str, default="./data/msmarco-passage/checkpoints/step-430000")
    # data args
    parser.add_argument("--tokenize_dir", type=str, default="./data/trec-dl-2019/tokenize")
    parser.add_argument("--embedding_dir", type=str, default="./data/msmarco-passage/avg_L2_CE_lr1e-5/embeddings/dense_retriever/step-340000")
    parser.add_argument("--index_dir", type=str, default="./data/trec-dl-2019/avg_L2_CE_lr1e-5/index")
    parser.add_argument("--retrieved_result_dir", type=str, default="./data/trec-dl-2019/avg_L2_CE_lr1e-5/results")
    # inference args
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--queries_batch_size", type=int, default=100)
    parser.add_argument("--hit_num", type=int, default=1000)
    parser.add_argument("--max_query_length", type=int, default=32)
    # ivf args
    parser.add_argument("--center_nums", type=int, default=3000)
    parser.add_argument("--nprobe", type=int, default=10)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.do_build:
        build_faiss_index(args)
    if args.do_retrieve:
        retrieve_from_index(args)


if __name__ == "__main__":
    main()