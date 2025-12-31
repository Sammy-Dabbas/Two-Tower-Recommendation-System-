import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from data_loader import load_movielens_with_features
from sklearn.preprocessing import OneHotEncoder
import pandas as pd



#Random seeds
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

    
class BPRDataset(Dataset):
      def __init__(self, df, num_items, item_feature_lookup):
        positive_df = df[df['rating'] >= 4].copy()
        self.users = positive_df['user_id'].values - 1
        self.pos_items = positive_df['item_id'].values - 1 
        self.num_items = num_items

        self.user_features = positive_df[['age', 'gender', 'occupation']].values 
        item_genres = ['unknown', 'Action', 'Adventure', 'Animation', 'Children',
                        'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                        'Sci-Fi', 'Thriller', 'War', 'Western'] 
        self.pos_item_features = positive_df[item_genres].values
        item_df = df[['item_id'] + item_genres].drop_duplicates('item_id')
        item_df = item_df.sort_values('item_id')
        self.item_feature_lookup = item_feature_lookup

      def __len__(self): 
            return len(self.users)

      def __getitem__(self, idx):
          user = self.users[idx]           #User who liked something
          pos_item = self.pos_items[idx]   #Item they liked

          #Sample one random negative item
          neg_item = np.random.randint(0, self.num_items)
          user_features = self.user_features[idx]
          pos_item_feat = self.pos_item_features[idx]
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
      def __init__(self, num_users, num_genders, num_occupations, embedding_dim=50, gender_embedding_dim = 4, occupation_embedding_dim = 12, hidden_dim=128):
          super().__init__()
          total_dims = embedding_dim + gender_embedding_dim + occupation_embedding_dim + 1
          self.user_embedding = nn.Embedding(num_users, embedding_dim)
          self.gender_embedding = nn.Embedding(num_genders, gender_embedding_dim)
          self.occupations_embedding = nn.Embedding(num_occupations, occupation_embedding_dim)
          self.fc1 = nn.Linear(total_dims, hidden_dim)  
          self.relu = nn.ReLU()            #Make negatives zero
          self.fc2 = nn.Linear(hidden_dim, hidden_dim)  
          
          nn.init.xavier_normal_(self.user_embedding.weight.data)
      def forward(self, user_ids, user_features): 
          user_emb = self.user_embedding(user_ids)
          age = user_features[:, 0:1]  #First column, keep 2D
          gender_idx = user_features[:, 1].long()  # Second column, convert to long
          occupation_idx = user_features[:, 2].long()
          gender_emb = self.gender_embedding(gender_idx)
          occupation_emb = self.occupations_embedding(occupation_idx)
          age_norm = age / 100.0
          x = torch.cat([user_emb, gender_emb, occupation_emb, age_norm], dim=1)
          x = self.fc1(x)
          x = self.relu(x)
          x = self.fc2(x)

          return x 
      
class ItemTower(nn.Module):
      def __init__(self, num_items, embedding_dim=50, hidden_dim=128, output_dim=128):
          super().__init__()

        
          self.item_embedding = nn.Embedding(num_items, embedding_dim)
           
          self.fc1 = nn.Linear(69, hidden_dim)
          self.fc2 = nn.Linear(hidden_dim, output_dim)
          self.relu = nn.ReLU()

          
          nn.init.xavier_normal_(self.item_embedding.weight.data)

      def forward(self, item_ids, item_features):
          item_emb = self.item_embedding(item_ids)
          x = torch.cat([item_emb, item_features], dim=1)
          x = self.fc1(x)
          x = self.relu(x)
          x = self.fc2(x)
          return x
      

class TwoTowerModel(nn.Module):
      def __init__(self, num_users, num_items, num_genders, num_occupations, embedding_dim=50, hidden_dim=128, output_dim=128):
          super().__init__()

          #Create both towers
          self.user_tower = UserTower(num_users, num_genders, num_occupations, embedding_dim, hidden_dim)
          self.item_tower = ItemTower(num_items, embedding_dim, hidden_dim, output_dim)

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

def train_model(model, train_data, num_epochs=10): 
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.000)
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

def recall_at_k(model, test_df, train_df, item_feature_lookup, k=10):
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
            train_items = len(test_df)
            if len(all_items) == 0 or len(liked_items) == 0:
                continue

        
            #creating item and user tensors
            all_items_list = list(all_items)
            num_unseen = len(all_items_list)
            user_features = test_df[test_df['user_id'] == user_id][['age', 'gender', 'occupation']].iloc[0].values
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




df = load_movielens_with_features()
df = df.sort_values('timestamp')
genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Children',
                'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                'Sci-Fi', 'Thriller', 'War', 'Western']
item_lookup_df = df[['item_id'] + genre_cols].drop_duplicates('item_id').sort_values('item_id')
item_feature_lookup = item_lookup_df[genre_cols].values
# encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
# categorical_columns = ["gender"]

# one_hot = encoder.fit_transform(df[categorical_columns])

# one_hot_df = pd.DataFrame(
#     one_hot,
#     columns=encoder.get_feature_names_out(categorical_columns),
#     index=df.index,
# ).astype(int)
# df = pd.concat([df, one_hot_df], axis=1)
# df = df.drop('gender', axis=1)
split_idx = int(len(df) * 0.8)
train_df = df[:split_idx]
test_df = df[split_idx:]




num_items = df['item_id'].max()
num_users = df['user_id'].max()

num_genders = df['gender'].nunique()
num_occupations = df['occupation'].nunique()



train_dataset = BPRDataset(train_df, num_items, item_feature_lookup)
test_dataset = BPRDataset(test_df, num_items, item_feature_lookup)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True) 




# print("Training Matrix Factorization Baseline")
# baseline_model = MatrixFactorization(num_users, num_items, embedding_dim=50)
# train_model(baseline_model, train_loader, num_epochs=10)
# evaluate_bpr(baseline_model, test_loader)
# baseline_recall = recall_at_k(baseline_model, test_df, train_df, k=10)
# print(f"Baseline Recall@10: {baseline_recall:.4f}")

print("Training Two-Tower Model")
two_tower_model = TwoTowerModel(num_users, num_items, num_genders, num_occupations, embedding_dim=50, hidden_dim=128, output_dim=128)
train_model(two_tower_model, train_loader, num_epochs=10)
evaluate_bpr(two_tower_model, test_loader)
two_tower_recall = recall_at_k(two_tower_model, test_df, train_df, item_feature_lookup, k=10)
print(f"Two-Tower Recall@10: {two_tower_recall:.4f}")


