import torch
import numpy as np
import faiss
from matrix_factorization import TwoTowerModel
from data_loader import load_movielens_with_features

df = load_movielens_with_features()
num_items = df['item_id'].max()
num_users = df['user_id'].max()
num_genders = df['gender'].nunique()
num_occupations = df['occupation'].nunique()
print("loading model")
model = TwoTowerModel(num_users, num_items, num_genders, num_occupations, embedding_dim=50, hidden_dim=128, output_dim=128)

model.load_state_dict(torch.load("two_tower_model.pth"))
model.eval()
print("Model loaded")

print("loading faiss index.")
index = faiss.read_index("item_index.faiss")
print(f"Index loaded with {index.ntotal} items")

def get_recommendations(user_id, k=10): 
    with torch.no_grad(): 
          user_emb = model.user_tower(torch.tensor([user_id]), torch.tensor([df[df['user_id'] == user_id][['age', 'gender', 'occupation']].iloc[0].values])).cpu().numpy()
    scores, item_ids = index.search(user_emb, k=k)
    return item_ids[0], scores[0]

user_id = 196
print("Top 10 for user {user_id}:")
items, scores = get_recommendations(user_id, k=10)
print(f"items: {items}")
print(f"scores: {scores}")
