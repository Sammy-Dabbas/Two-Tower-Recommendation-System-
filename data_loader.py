import pandas as pd

DATA_FILE_PATH = 'data/ml-100k/u.data'

def load_movielens(filepath): 
    df = pd.read_csv(filepath, 
                    sep='\t', 
                    names=['user_id', 'item_id', 'rating', 'timestamp'])
    return df 

def load_user_features(filepath='data/ml-100k/u.user'):
      """Load user demographic features from u.user file"""
      user_df = pd.read_csv(filepath,
                           sep='|',
                           names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
      #Drop zip_code as it's not useful for recommendations, we can encode to states later.
      user_df = user_df.drop('zip_code', axis=1)
      return user_df

def load_item_features(filepath='data/ml-100k/u.item'):
      """Load item genre features from u.item file"""
      genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Children',
                    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                    'Sci-Fi', 'Thriller', 'War', 'Western']

      columns = ['item_id', 'title', 'release_date', 'video_release', 'url'] + genre_cols

      item_df = pd.read_csv(filepath,
                           sep='|',
                           names=columns,
                           encoding='latin-1')

      # Keep only item_id and genre columns
      item_df = item_df[['item_id'] + genre_cols]
      return item_df

def load_movielens_with_features():
      """Load MovieLens data with user and item features merged"""
      # Load base ratings data
      df = load_movielens(DATA_FILE_PATH)

      # Load feature data
      user_df = load_user_features()
      item_df = load_item_features()

      # Merge user features
      df = pd.merge(df, user_df, on='user_id', how='left')
      df['gender'] = df['gender'].astype('category').cat.codes  
      df['occupation'] = df['occupation'].astype('category').cat.codes  
      # Merge item features
      df = pd.merge(df, item_df, on='item_id', how='left')
      print(df)
      return df


if __name__ == "__main__":
      df = load_movielens_with_features()
      print(f"Shape: {df.shape}")
      print(f"Columns: {df.columns.tolist()}")
      print(df.head())