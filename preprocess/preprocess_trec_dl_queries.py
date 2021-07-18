import argparse
import os
import json
import shutil
import logging
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s-%(levelname)s-%(name)s- %(message)s",
    datefmt="%d %H:%M:%S",
    level=logging.INFO
)


def preprocess_trec_dl_queries(args):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # tokenize queries in train set
    if not os.path.exists(args.tokenize_dir):
        os.makedirs(args.tokenize_dir)
    queries_dev_dir = os.path.join(args.official_data_dir, "msmarco-test{}-queries.tsv".format(args.year))
    tokenized_queries_dev_dir = os.path.join(args.tokenize_dir, "tokenized_queries.test.json")
    with open(queries_dev_dir) as fin, open(tokenized_queries_dev_dir, "w") as fout:
        for line in fin:
            query_id, query_content = line.split("\t")
            token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query_content))
            fout.write(json.dumps({
                "id": int(query_id),
                "ids": token_ids,
            }) + "\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official_data_dir", type=str, default="./data/trec-dl-2020/official_data")
    parser.add_argument("--tokenize_dir", type=str, default="./data/trec-dl-2020/tokenize")
    parser.add_argument("--year", type=str, default="2020")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logger.info(args)
    preprocess_trec_dl_queries(args)


if __name__ == "__main__":
    main()