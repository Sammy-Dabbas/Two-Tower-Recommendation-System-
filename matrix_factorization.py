import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from config.config import load_config
from loaders.factory import get_data_loader
  # Load config


config = load_config()
data = get_data_loader(config)


print(f"Dataset: {config['dataset']['name']}\n")





#Random seed
SEED = config['training']['random_seed']
np.random.seed(SEED)
torch.manual_seed(SEED)

batch_size = config['training']['batch_size']


class BPRDataset(Dataset):
      def __init__(self, df, num_items, user_feature_lookup, item_feature_lookup):
        positive_df = df[df['rating'] >= 4].copy()
        self.users = positive_df['user_id'].values - 1
        self.pos_items = positive_df['item_id'].values - 1 
        self.num_items = num_items

        
        self.item_feature_lookup = item_feature_lookup
        self.user_feature_lookup = user_feature_lookup

      def __len__(self): 
            return len(self.users)

      def __getitem__(self, idx):
          user = self.users[idx]           #User who liked something
          pos_item = self.pos_items[idx]   #Item they liked

          #Sample one random negative item
          neg_item = np.random.randint(0, self.num_items)
          user_features = self.user_feature_lookup[user]
          pos_item_feat = self.item_feature_lookup[pos_item]
          neg_item_feat = self.item_feature_lookup[neg_item]

          return (
          torch.tensor(user, dtype=torch.long),
          torch.tensor(pos_item, dtype=torch.long),
          torch.tensor(neg_item, dtype=torch.long),
          torch.tensor(user_features, dtype=torch.float32),
          torch.tensor(pos_item_feat, dtype=torch.float32),
          torch.tensor(neg_item_feat, dtype=torch.float32)
      )

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

def train_model(model, train_data, config):

    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    bpr_loss = BPRLoss()
    for epoch in range(num_epochs): 
        for user_ids, pos_items, neg_items, user_feat, pos_item_feat, neg_item_feat in train_data: 
            pos_scores  = model(user_ids, pos_items, user_feat, pos_item_feat)
            neg_scores = model(user_ids, neg_items, user_feat, neg_item_feat)
            
            loss = bpr_loss(pos_scores, neg_scores)
            

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        print(f"Epoch {epoch + 1}, loss: {loss.item():.4f}")

def evaluate_bpr(model, test_data):
      model.eval()

      pos_scores_list = []
      neg_scores_list = []

      with torch.no_grad():
         for user_ids, pos_items, neg_items, user_feat, pos_item_feat, neg_item_feat in test_data:              
              pos_scores  = model(user_ids, pos_items, user_feat, pos_item_feat)
              neg_scores = model(user_ids, neg_items, user_feat, neg_item_feat)

              pos_scores_list.append(pos_scores.mean().item())
              neg_scores_list.append(neg_scores.mean().item())

      avg_pos = sum(pos_scores_list) / len(pos_scores_list)
      avg_neg = sum(neg_scores_list) / len(neg_scores_list)

      print(f"Avg positive score: {avg_pos:.4f}")
      print(f"Avg negative score: {avg_neg:.4f}")
      print(f"Difference: {avg_pos - avg_neg:.4f}")

def recall_at_k(model, test_df, train_df, num_items, user_features_lookup, item_feature_lookup, k=10):
    """
    Compute Recall@K: Of all items a user actually likes, what percentage appear in top-K recommendations
    """
    model.eval()
    recalls = []

    #Get the users who have liked items in test set where rating >= 4
    test_liked = test_df[test_df['rating'] >= 4].groupby('user_id')['item_id'].apply(set).to_dict()

    #Get items each user saw during training
    train_seen = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
   
    with torch.no_grad():
        for user_id, liked_items in test_liked.items():
            #Items this user saw in training 
            seen_items = train_seen.get(user_id, set())

            #All items we could recommend 
            all_items = set(range(1, num_items + 1)) - seen_items
            if len(all_items) == 0 or len(liked_items) == 0:
                continue

        
            #creating item and user tensors
            all_items_list = list(all_items)
            num_unseen = len(all_items_list)
            user_features = user_features_lookup[user_id - 1]
            user_tensor = torch.tensor([user_id - 1] * num_unseen, dtype=torch.long)
            user_features_tensor = torch.tensor([user_features] * num_unseen, dtype=torch.float32)
            item_tensor = torch.tensor([item - 1 for item in all_items_list], dtype=torch.long)
          
           
            item_features_list = []
            for item in all_items_list: 
                features_for_this_item = item_feature_lookup[item - 1]
              
                item_features_list.append(features_for_this_item)
            item_features_tensor = torch.tensor(item_features_list, dtype=torch.float32)
            #get scores 
           
            scores  = model(user_tensor, item_tensor, user_features_tensor, item_features_tensor)
        
            k_actual = min(k, len(scores))
            top_scores, top_indices = torch.topk(scores, k_actual)
            top_items = [all_items_list[i] for i in top_indices.tolist()]
          
            #Caculate number of matched between top items and liked items
            matches = set(top_items) & liked_items

            recall = len(matches) / len(liked_items)
            recalls.append(recall)

            

    #Return average recall across all users
    return sum(recalls) / len(recalls) if recalls else 0.0




df = data['interactions']
df = df.sort_values('timestamp')


split_idx = int(len(df) * 0.8)
train_df = df[:split_idx]
test_df = df[split_idx:]




num_items = data['num_items']
num_users = data['num_users']



item_features_lookup = data['item_features']
user_features_lookup = data['user_features']

train_dataset = BPRDataset(train_df, num_items, user_features_lookup, item_features_lookup)
test_dataset = BPRDataset(test_df, num_items, user_features_lookup, item_features_lookup)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True) 




# print("Training Matrix Factorization Baseline")
# baseline_model = MatrixFactorization(num_users, num_items, embedding_dim=50)
# train_model(baseline_model, train_loader, num_epochs=10)
# evaluate_bpr(baseline_model, test_loader)
# baseline_recall = recall_at_k(baseline_model, test_df, train_df, k=10)
# print(f"Baseline Recall@10: {baseline_recall:.4f}")

user_feature_config = data['user_feature_config']
item_feature_config = data['item_feature_config']

print("Training Two-Tower Model")
two_tower_model = TwoTowerModel(num_users, num_items, user_feature_config, item_feature_config , config)
train_model(two_tower_model, train_loader, config)
evaluate_bpr(two_tower_model, test_loader)
two_tower_recall = recall_at_k(two_tower_model, test_df, train_df, num_items, user_features_lookup, item_features_lookup, k=10)
print(f"Two-Tower Recall@10: {two_tower_recall:.4f}")


