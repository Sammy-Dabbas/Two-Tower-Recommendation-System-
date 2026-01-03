import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from config.config import load_config
from models.two_tower import TwoTowerModel, BPRLoss
from loaders.factory import get_data_loader
from training import train_model
from evaluation import evaluate_bpr, recall_at_k
from loaders.dataset import BPRDataset
# Load config
config = load_config()
data = get_data_loader(config)


print(f"Dataset: {config['dataset']['name']}\n")


#Random seed
SEED = config['training']['random_seed']
np.random.seed(SEED)
torch.manual_seed(SEED)

batch_size = config['training']['batch_size']


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
torch.save(two_tower_model.state_dict(), "two_tower_model.pth")
evaluate_bpr(two_tower_model, test_loader)
two_tower_recall = recall_at_k(two_tower_model, test_df, train_df, num_items, user_features_lookup, item_features_lookup, k=10)
print(f"Two-Tower Recall@10: {two_tower_recall:.4f}")
