import pandas as pd
import numpy as np 



genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Children',
                    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                    'Sci-Fi', 'Thriller', 'War', 'Western']
                    
def load_movielens_100k(dataset_config):
      """
      Load MovieLens 100k with preprocessed features.

      Args:
          dataset_config: Dictionary from config['dataset']

      Returns:
          Dictionary with preprocessed features
      """
      data_path = dataset_config['data_path']
      user_path = dataset_config['user_path']
      item_path = dataset_config['item_path']

      df = pd.read_csv(data_path, 
                    sep='\t', 
                    names=['user_id', 'item_id', 'rating', 'timestamp'])


      """Load user demographic features from u.user file"""
      user_df = pd.read_csv(user_path,
                           sep='|',
                           names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
      #Drop zip_code as it's not useful for recommendations, we can encode to states later.
      user_df = user_df.drop('zip_code', axis=1)

      """Load item genre features from u.item file"""
      
      columns = ['item_id', 'title', 'release_date', 'video_release', 'url'] + genre_cols

      item_df = pd.read_csv(item_path,
                           sep='|',
                           names=columns,
                           encoding='latin-1')

      # Keep only item_id and genre columns
      item_df = item_df[['item_id'] + genre_cols]
      item_lookup_df = item_df.drop_duplicates('item_id').sort_values('item_id')
      item_features_numpy = item_lookup_df[genre_cols].values  # Just the 19 genre columns
      df = pd.merge(df, user_df, on='user_id', how='left')
 
      # Merge item features
      df = pd.merge(df, item_df, on='item_id', how='left')
      user_lookup_df = user_df.drop_duplicates('user_id').sort_values('user_id')

      age_norm_array = (user_lookup_df['age'].values / 100.0).reshape(-1, 1)
      gender_onehot_array = pd.get_dummies(user_lookup_df['gender']).values
      occupation_onehot_array = pd.get_dummies(user_lookup_df['occupation']).values
      
      user_features_numpy = np.concatenate([age_norm_array, gender_onehot_array, occupation_onehot_array], axis=1)
 
      num_items = df['item_id'].max()
      num_users = df['user_id'].max()

    
      return { 
            "interactions": df,
            "user_features": user_features_numpy,
            "item_features": item_features_numpy, 
            "user_features_dim": 24,
            "item_features_dim": 19,
            "num_users": num_users,
            "num_items": num_items
      }