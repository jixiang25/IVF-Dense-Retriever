import os


def output_retrieved_ranking_result(retrieved_result_dir, retrieved_result, hit_num):
    if not os.path.exists(retrieved_result_dir):
        os.makedirs(retrieved_result_dir)
    score_file = os.path.join(retrieved_result_dir, "top{}.score.txt".format(hit_num))
    rank_file = os.path.join(retrieved_result_dir, "top{}.rank.txt".format(hit_num))
    with open(score_file, "w") as fout_score, open(rank_file, "w") as fout_rank:
        for query_id, query_pq in retrieved_result.items():
            if query_pq.qsize() != hit_num:
                raise ValueError("Query {} should have {} hits".format(query_id, hit_num))
            for i in range(hit_num):
                doc_score, doc_id = query_pq.get_nowait()
                fout_score.write("{}\t{}\t{}\n".format(query_id, doc_id, doc_score))
                fout_rank.write("{}\t{}\t{}\n".format(query_id, doc_id, hit_num - i))