import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from data_loader import load_movielens


DATA_FILE_PATH = 'data/ml-100k/u.data'



    
class BPRDataset(Dataset):
      def __init__(self, df, num_items):
        positive_df = df[df['rating'] >= 4].copy()
        self.users = positive_df['user_id'].values - 1
        self.pos_items = positive_df['item_id'].values - 1 
        self.num_items = num_items


      def __len__(self): 
            return len(self.users)

      def __getitem__(self, idx):
          user = self.users[idx]           #User who liked something
          pos_item = self.pos_items[idx]   #Item they liked

          #Sample one random negative item
          neg_item = np.random.randint(0, self.num_items)

          return (
          torch.tensor(user, dtype=torch.long),
          torch.tensor(pos_item, dtype=torch.long),
          torch.tensor(neg_item, dtype=torch.long)
      )

class MatrixFactorization(nn.Module): 
    def __init__(self, num_users, num_items, embedding_dim=50): 
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight.data)
        nn.init.xavier_normal_(self.item_embedding.weight.data)

    def forward(self, user_ids, item_ids): 
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        scores = (user_emb * item_emb).sum(dim=1)
        return scores
    
class BPRLoss(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, pos_scores, neg_scores): 
        x_uij = pos_scores - neg_scores
        loss = -F.logsigmoid(x_uij).mean()
        return loss 

def train_model(model, train_data, num_epochs=10): 
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.000)
    bpr_loss = BPRLoss()
    for epoch in range(num_epochs): 
        for user_ids, pos_items, neg_items in train_data: 
            pos_scores  = model(user_ids, pos_items)
            neg_scores = model(user_ids, neg_items)
            
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
          for user_ids, pos_items, neg_items in test_data:
              pos_scores = model(user_ids, pos_items)
              neg_scores = model(user_ids, neg_items)

              pos_scores_list.append(pos_scores.mean().item())
              neg_scores_list.append(neg_scores.mean().item())

      avg_pos = sum(pos_scores_list) / len(pos_scores_list)
      avg_neg = sum(neg_scores_list) / len(neg_scores_list)

      print(f"Avg positive score: {avg_pos:.4f}")
      print(f"Avg negative score: {avg_neg:.4f}")
      print(f"Difference: {avg_pos - avg_neg:.4f}")

        


df = load_movielens(DATA_FILE_PATH)
df = df.sort_values('timestamp')

split_idx = int(len(df) * 0.8)
train_df = df[:split_idx]
test_df = df[split_idx:]
num_items = df['item_id'].max()
num_users = df['user_id'].max()

train_dataset = BPRDataset(train_df, num_items)
test_dataset = BPRDataset(test_df, num_items)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True) 


model = MatrixFactorization(num_users, num_items, embedding_dim=50)
train_model(model, train_loader, num_epochs = 10) 
evaluate_bpr(model, test_loader)