import torch
import numpy as np
import faiss
from matrix_factorization import TwoTowerModel
import time 
def compute_item_embeddings(model, num_items):
      """
      Extract embeddings for ALL items from trained model.
      """
      model.eval()
      with torch.no_grad():
        all_item_ids = torch.arange(0, num_items, dtype = torch.long)
        item_embeddings = model.item_tower(all_item_ids)
      return item_embeddings.cpu().numpy()
def build_faiss_index(item_embeddings): 
    embedding_dim = item_embeddings.shape[1]  # 128
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(item_embeddings.astype('float32'))
    return index




if __name__ == "__main__":
      from matrix_factorization import load_movielens, BPRDataset, train_model
      from torch.utils.data import DataLoader

      # Quick training for testing
      df = load_movielens('data/ml-100k/u.data')
      train_df = df[:int(len(df)*0.8)]
      num_items = df['item_id'].max()
      num_users = df['user_id'].max()

      train_dataset = BPRDataset(train_df, num_items)
      train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

      model = TwoTowerModel(num_users, num_items, embedding_dim=50, hidden_dim=128, output_dim=128)
      train_model(model, train_loader, num_epochs=10)  # Just 3 for testing

      print("Computing item embeddings")
      embeddings = compute_item_embeddings(model, num_items)
      print(f"Shape: {embeddings.shape}")
      index = build_faiss_index(embeddings)
      start_brute = time.time()
      with torch.no_grad():
            #Create tensors for ALL items
            user_id = 0
            user_repeated = torch.tensor([user_id] * num_items, dtype=torch.long)
            all_items = torch.arange(0, num_items, dtype=torch.long)

            #score ALL items
            scores = model(user_repeated, all_items)

            #Get top 10
            top_scores, top_indices = torch.topk(scores, k=10)
      brute_time = (time.time() - start_brute) * 1000
      print(f"FAISS time: {brute_time:.2f}ms")

      start_faiss = time.time()
    
      with torch.no_grad():
        user_emb = model.user_tower(torch.tensor([0])).cpu().numpy()
        scores, item_ids = index.search(user_emb, k=10)
        faiss_time = (time.time() - start_faiss) * 1000
        print(f"FAISS time: {faiss_time:.2f}ms")
        print(f"Top 10 items: {item_ids[0]}")
     
    

