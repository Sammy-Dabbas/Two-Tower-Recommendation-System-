import torch
import torch.nn as nn


class MatrixFactorization(nn.Module): 
    def __init__(self, num_users, num_items, embedding_dim=50): 
        super().__init__()
  
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.xavier_normal_(self.user_embedding.weight.data)
        nn.init.xavier_normal_(self.item_embedding.weight.data)

        nn.init.zeros_(self.user_bias.weight.data)
        nn.init.zeros_(self.item_bias.weight.data)

    def forward(self, user_ids, item_ids): 
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        scores = (user_emb * item_emb).sum(dim=1)

        scores = scores + self.user_bias(user_ids).squeeze()
        scores = scores + self.item_bias(item_ids).squeeze()
        scores = scores + self.global_bias

        return scores
    