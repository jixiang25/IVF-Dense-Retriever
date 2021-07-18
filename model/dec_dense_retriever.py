import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import BertModel, BertPreTrainedModel
from model.dual_encoder import _average_sequence_embeddings
from utils.similarity_functions import l2_distance


class IvfDenseRetriever(BertPreTrainedModel):
    def __init__(self, config):
        super(IvfDenseRetriever, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()
        self.repr_type = config.repr_type
        self.repr_normalized = config.repr_normalized
        self.nearest_centroid_num = config.nearest_centroid_num

    def set_centroid_embedding_layer(self, centroid_embedding_dir, cluster_num, hidden_size):
        centroid_embedding_memmap = np.memmap(
            os.path.join(centroid_embedding_dir, "centroid_embeddings.memmap"),
            dtype="float32",
            mode="c",
            shape=(cluster_num, hidden_size)
        )
        centroid_embedding_tensor = torch.from_numpy(centroid_embedding_memmap)
        self.centroid_embeddings = nn.Parameter(centroid_embedding_tensor)

    def get_centroid_embeddings(self):
        return self.centroid_embeddings

    def forward(self, input_ids, attention_mask):
        sequence_output, cls_embeddings = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # choose representation via `repr_type`
        if self.repr_type == "avg":
            text_embedding = _average_sequence_embeddings(
                sequence_output=sequence_output,
                mask=attention_mask
            )
        else:
            text_embedding = cls_embeddings
        # execuate normalization if `is_normalize` is true
        if self.repr_normalized:
            text_embedding = F.normalize(text_embedding, dim=1)
        return text_embedding


class MultiLabelCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MultiLabelCrossEntropyLoss, self).__init__()

    def forward(self, input, target):
        batch_size, centroid_num = input.size()
        input_exp = torch.exp(input)
        device = input.device

        mask = torch.zeros((batch_size, centroid_num)).to(device).scatter_(1, target, 1)
        p = torch.sum(input_exp * mask, dim=-1)
        q = torch.sum(input_exp, dim=-1)

        loss = torch.sum(torch.log(q) - torch.log(p), dim=-1) / batch_size
        return loss
        