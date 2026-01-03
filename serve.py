import torch
import numpy as np
import faiss
from matrix_factorization import TwoTowerModel
from config.config import load_config
from loaders.factory import get_data_loader
  # Load config


config = load_config()
data = get_data_loader(config)

df = data['interactions']


print("loading model")
item_features_lookup = data['item_features']
user_features_lookup = data['user_features']
user_feature_dim = data['user_features_dim']
item_feature_dim = data['item_features_dim']
num_items = df['item_id'].max()
num_users = df['user_id'].max()
model = TwoTowerModel(num_users, num_items, user_feature_dim, item_feature_dim, config)

model.load_state_dict(torch.load("two_tower_model.pth"))
model.eval()
print("Model loaded")

print("loading faiss index.")
index = faiss.read_index("item_index.faiss")
print(f"Index loaded with {index.ntotal} items")

def get_recommendations(user_id, k=10): 
    with torch.no_grad(): 
          user_emb = model.user_tower(torch.tensor([196]), torch.tensor([user_features_lookup[user_id - 1]], dtype=torch.float32)).cpu().numpy()
    scores, item_ids = index.search(user_emb, k=k)
    return item_ids[0], scores[0]

user_id = 196
print(f"Top 10 for user {user_id}:")
items, scores = get_recommendations(user_id, k=10)
print(f"items: {items}")
print(f"scores: {scores}")
