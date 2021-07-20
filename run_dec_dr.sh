#!/bin/bash

python -m train.train_dec_dr \
    --collection_memmap_dir=./data/msmarco-passage/avg_L2_CE_lr1e-5/collection_memmap/complete \
    --tokenize_dir=./data/msmarco-passage/avg_L2_CE_lr1e-5/tokenize \
    --tripplets_and_qrels_dir=./data/msmarco-passage/avg_L2_CE_lr1e-5/tripplets_and_qrels \
    --centroid_embedding_dir=./data/msmarco-passage/avg_L2_CE_lr1e-5/centroid_embeddings \
    --max_query_length=32 \
    --max_doc_length=256 \
    --device=cpu \
    --train_epochs=1 \
    --gradient_accumulate_steps=2 \
    --batch_size_per_gpu=20 \
    --learning_rate=3e-6 \
    --warmup_steps=10000 \
    --load_model_path=bert-base-uncased \
    --repr_type=avg \
    --similarity_type=L2 \
    --loss_type=cross-entropy \
    --cluster_num=1000 \
    --checkpoint_dir=./data/msmarco-passage/avg_L2_CE_lr1e-5/checkpoints/dec-dr \
    --logging_dir=./data/msmarco-passage/avg_L2_CE_lr1e-5/log/dec-dr