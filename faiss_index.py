import torch
import numpy as np
import faiss
import time 
from data_loader import load_movielens_with_features



FILENAME = "item_index.faiss"
def compute_item_embeddings(model, num_items, item_feature_lookup):
      """
      Extract embeddings for ALL items from trained model.
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
 
      

      #Quick training for testing
      df = load_movielens_with_features()
      
      df = df.sort_values('timestamp')
      genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Children',
                'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                'Sci-Fi', 'Thriller', 'War', 'Western']
      item_lookup_df = df[['item_id'] + genre_cols].drop_duplicates('item_id').sort_values('item_id')
      item_feature_lookup = item_lookup_df[genre_cols].values
      train_df = df[:int(len(df)*0.8)]
      num_items = df['item_id'].max()
      num_users = df['user_id'].max()
      


      num_genders = df['gender'].nunique()
      num_occupations = df['occupation'].nunique()
      

    
      from matrix_factorization import BPRDataset, train_model
      from torch.utils.data import DataLoader
      from matrix_factorization import TwoTowerModel
      train_dataset = BPRDataset(train_df, num_items, item_feature_lookup)
      train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
      

      model = TwoTowerModel(num_users, num_items, num_genders, num_occupations, embedding_dim=50, hidden_dim=128, output_dim=128)
      train_model(model, train_loader, num_epochs=10)  

      print("Computing item embeddings")
      embeddings = compute_item_embeddings(model, num_items, item_feature_lookup)
      print(f"Shape: {embeddings.shape}")
      index = build_faiss_index(embeddings)
     
     

      start_faiss = time.time()
    
      with torch.no_grad():
        user_emb = model.user_tower(torch.tensor([196]), torch.tensor([df[df['user_id'] == 196][['age', 'gender', 'occupation']].iloc[0].values], dtype=torch.float32)).cpu().numpy()
        scores, item_ids = index.search(user_emb, k=10)
        faiss_time = (time.time() - start_faiss) * 1000
        print(f"FAISS time: {faiss_time:.2f}ms")
        print(f"Top 10 items: {item_ids[0]}")

      saveIndex(index)
      torch.save(model.state_dict(), "two_tower_model.pth")
     
    

