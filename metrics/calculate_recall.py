import os
import argparse


def calculate_recall(args):
    candidates_dict, qrels_dict = {}, {}
    with open(args.candidates_dir) as fin:
        for line in fin:
            qid, pid, _, _ = line.split("\t")
            qid, pid = int(qid), int(pid)
            if qid not in candidates_dict:
                candidates_dict[qid] = set()
            if len(candidates_dict[qid]) == 100:
                continue
            candidates_dict[qid].add(pid)

    recall_dict = {}
    tot, hit = 0, 0
    with open(args.qrels_dir) as fin:
        for line in fin:
            qid, _, pid, rel = line.split()
            qid, pid, rel = int(qid), int(pid), int(rel)
            if qid not in recall_dict:
                recall_dict[qid] = {
                    "hit": 0,
                    "tot": 0,
                    "recall": 0
                }
            if rel > 0:
                tot += 1
                recall_dict[qid]["tot"] += 1
                if pid in candidates_dict[qid]:
                    hit += 1
                    recall_dict[qid]["hit"] += 1
    
    for idx, qid in enumerate(recall_dict):
        recall_dict[qid]["recall"] = recall_dict[qid]["hit"] / recall_dict[qid]["tot"]
        print("{}\t{}\t{}\t{}\t{:.6}".format(
            idx + 1,
            qid,
            recall_dict[qid]["hit"],
            recall_dict[qid]["tot"],
            recall_dict[qid]["recall"]
        ))
    print("recall = {} / {} = {}".format(hit, tot, hit / tot))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels_dir", type=str, default="./data/trec-dl-2019/official_data/2019qrels-pass.txt")
    parser.add_argument("--candidates_dir", type=str, default="./data/trec-dl-2019/avg_L2_CE_lr1e-5/results/top1000.tsv")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    calculate_recall(args)


if __name__ == "__main__":
    main()

