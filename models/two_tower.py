import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
      def __init__(self, num_users, user_feature_config,  embedding_dim, hidden_dim):
          super().__init__()
          self.user_embedding = nn.Embedding(num_users, embedding_dim)
          self.feature_config = user_feature_config
          self.feature_embedding = nn.ModuleDict()
          total_feature_dim = 0
          for feat in user_feature_config: 
              if feat['type'] == 'embedding': 
                  self.feature_embedding[feat['name']] = nn.Embedding(feat['vocab_size'], feat['embedding_dim'])
                  total_feature_dim += feat['embedding_dim']
              elif (feat['type'] == 'continuous'):
                  total_feature_dim += 1
              elif (feat['type'] == 'multi_hot'):
                  total_feature_dim += feat['size'] 

          

          self.fc1 = nn.Linear(embedding_dim + total_feature_dim, hidden_dim)  
          self.relu = nn.ReLU()            #Make negatives zero
          self.fc2 = nn.Linear(hidden_dim, hidden_dim)  
          
          nn.init.xavier_normal_(self.user_embedding.weight.data)

      def forward(self, user_ids, user_features): 
          user_emb = self.user_embedding(user_ids)
          processed = []
          for feat in self.feature_config: 
              if feat['type'] == 'embedding': 
                  idx = user_features[:, feat['position']].long()
                  emb = self.feature_embedding[feat['name']](idx)
                  processed.append(emb) 
              elif feat['type'] == 'continuous': 
                  value = user_features[:, feat['position']:feat['position'] + 1]
                  processed.append(value)
              elif feat['type'] == 'multihot': 
                  start = feat['position']
                  end = start + feat['size']
                  value = user_features[:, start:end]
                  processed.append(value)
          
          feature_vec = torch.cat(processed, dim =1)
          x = torch.cat([user_emb, feature_vec], dim=1)
          x = self.fc1(x)
          x = self.relu(x)
          x = self.fc2(x)

          return x 
      
class ItemTower(nn.Module):
      def __init__(self, num_items, item_feature_config, embedding_dim, hidden_dim, output_dim):
          super().__init__()
          self.feature_config = item_feature_config
          self.feature_embedding = nn.ModuleDict()
          total_feature_dim = 0

          for feat in self.feature_config: 
              if feat['type'] == 'embedding': 
                  self.feature_embedding[feat['name']] = nn.Embedding(feat['vocab_size'], feat['embedding_dim'])
                  total_feature_dim += feat['embedding_dim']
              elif (feat['type'] == 'continuous'):
                  total_feature_dim += 1
              elif (feat['type'] == 'multi_hot'):
                  total_feature_dim += feat['size'] 
          total_dims = embedding_dim + total_feature_dim
          self.item_embedding = nn.Embedding(num_items, embedding_dim)
           
          self.fc1 = nn.Linear(total_dims, hidden_dim)
          self.fc2 = nn.Linear(hidden_dim, output_dim)
          self.relu = nn.ReLU()

          
          nn.init.xavier_normal_(self.item_embedding.weight.data)

      def forward(self, item_ids, item_features):
          
          processed = []
          for feat in self.feature_config: 
              if feat['type'] == 'embedding': 
                  idx = item_features[:, feat['position']].long()
                  emb = self.feature_embedding[feat['name']](idx)
                  processed.append(emb) 
              elif feat['type'] == 'continuous': 
                  value = item_features[:, feat['position']:feat['position'] + 1]
                  processed.append(value)
              elif feat['type'] == 'multi_hot': 
                  start = feat['position']
                  end = start + feat['size']
                  value = item_features[:, start:end]
                  processed.append(value)
          feature_vec = torch.cat(processed, dim =1)
          item_emb = self.item_embedding(item_ids)
          x = torch.cat([item_emb, feature_vec], dim=1)
          x = self.fc1(x)
          x = self.relu(x)
          x = self.fc2(x)
          return x
      

class TwoTowerModel(nn.Module):
      def __init__(self, num_users, num_items, user_feature_config, item_feature_config, config):
          super().__init__()
          embedding_dim = config['model']['embedding_dim']
          hidden_dim = config['model']['hidden_dim']
          output_dim = config['model']['output_dim']         
          #Create both towers
          self.user_tower = UserTower(num_users, user_feature_config, embedding_dim, hidden_dim)
          self.item_tower = ItemTower(num_items, item_feature_config, embedding_dim, hidden_dim, output_dim)

      def forward(self, user_ids, item_ids, user_features, item_features):
          #Get representations from both towers
          user_rep = self.user_tower(user_ids, user_features)
          item_rep = self.item_tower(item_ids, item_features)

          #Dot product 
          scores = (user_rep * item_rep).sum(dim=1)
          return scores
class BPRLoss(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, pos_scores, neg_scores): 
        x_uij = pos_scores - neg_scores
        loss = -F.logsigmoid(x_uij).mean()
        return loss 