import torch
import numpy as np
from torch.utils.data import Dataset


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