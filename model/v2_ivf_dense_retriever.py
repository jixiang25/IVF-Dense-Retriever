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

    def set_centroid_embedding_layer(self, centroid_embedding_dir, cluster_num, hidden_size, device):
        centroid_embedding_memmap = np.memmap(
            os.path.join(centroid_embedding_dir, "centroid_embeddings.memmap"),
            dtype="float32",
            mode="c",
            shape=(cluster_num, hidden_size)
        )
        self.centroid_embeddings = torch.from_numpy(centroid_embedding_memmap).to(device)
        self.centroid_embeddings.requires_grad = False

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


class ClusterMarginLoss(nn.Module):
    def __init__(self, nearest_centroid_num, cluster_num):
        super(ClusterMarginLoss, self).__init__()
        self.nearest_centroid_num = nearest_centroid_num
        self.cluster_num = cluster_num

    # def forward(self, input, target):
    #     batch_size = input.size()[0]
    #     device = input.device

    #     top_scores, _ = input.topk(k=self.nearest_centroid_num)
    #     last_chosen_score = top_scores[:, self.nearest_centroid_num - 1]

    #     mask = torch.ones((batch_size, self.cluster_num)).to(device) * -100000
    #     mask = mask.scatter_(1, target, 0)
    #     target_scores, _ = torch.topk(mask + input, k=1)
    #     target_scores = target_scores.squeeze(dim=-1)

    #     batch_loss = torch.maximum(last_chosen_score - target_scores, torch.zeros(batch_size).to(device))
    #     avg_loss = batch_loss.sum() / batch_size
    #     return avg_loss

    def forward(self, input, target):
        batch_size = input.size()[0]
        device = input.device
        num = self.cluster_num - self.nearest_centroid_num + 1

        bottom_scores, _ = input.topk(k=num, largest=False)

        mask = torch.ones((batch_size, self.cluster_num)).to(device) * -100000
        mask = mask.scatter_(1, target, 0)
        target_scores, _ = torch.topk(mask + input, k=1)

        batch_loss = torch.maximum(bottom_scores - target_scores, torch.zeros(batch_size, num).to(device))
        avg_loss = batch_loss.sum() / batch_size
        return avg_loss / 500