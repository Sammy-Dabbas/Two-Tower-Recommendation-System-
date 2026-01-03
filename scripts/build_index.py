import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import faiss
import time 
from config.config import load_config
from loaders.factory import get_data_loader
from models.two_tower import TwoTowerModel

# Load config
config = load_config()
data = get_data_loader(config)


FILENAME = "item_index.faiss"
def compute_item_embeddings(model, num_items, item_feature_lookup):
      """
      Extract embeddings for all items from trained model.
      """
      model.eval()
      with torch.no_grad():
        all_item_ids = torch.arange(0, num_items, dtype = torch.long)
        all_item_features = torch.tensor(item_feature_lookup, dtype=torch.float32)
        item_embeddings = model.item_tower(all_item_ids, all_item_features)

      return item_embeddings.cpu().numpy()
def build_faiss_index(item_embeddings): 
    embedding_dim = item_embeddings.shape[1]  # 128
    index = faiss.IndexFlatIP(embedding_dim)  #will use HNSW for larger datasets 
    index.add(item_embeddings.astype('float32'))
    return index

def saveIndex(index): 
    faiss.write_index(index, FILENAME)
    print(f"Index saved to {FILENAME}")

if __name__ == "__main__":
      
      df = data['interactions']
      df = df.sort_values('timestamp')
     
      train_df = df[:int(len(df)*0.8)]
      num_items = df['item_id'].max()
      num_users = df['user_id'].max()
      
      item_features_lookup = data['item_features']
      user_features_lookup = data['user_features']
      
      user_feature_config = data['user_feature_config']
      item_feature_config = data['item_feature_config']
      model = TwoTowerModel(num_users, num_items, user_feature_config, item_feature_config, config)
      model.load_state_dict(torch.load("two_tower_model.pth"))
      model.eval()
      


      print("Computing item embeddings")
      embeddings = compute_item_embeddings(model, num_items, item_features_lookup)
      print(f"Shape: {embeddings.shape}")
      index = build_faiss_index(embeddings)
     
     

      start_faiss = time.time()
      user_id = 196
      with torch.no_grad():
        user_emb = model.user_tower(torch.tensor([196]), torch.tensor([user_features_lookup[user_id - 1]], dtype=torch.float32)).cpu().numpy()
        scores, item_ids = index.search(user_emb, k=10)
        faiss_time = (time.time() - start_faiss) * 1000
        print(f"FAISS time: {faiss_time:.2f}ms")
        print(f"Top 10 items: {item_ids[0]}")

      saveIndex(index)
    
     
    

