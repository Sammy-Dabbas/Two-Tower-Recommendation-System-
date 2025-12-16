import pandas as pd

DATA_FILE_PATH = 'data/ml-100k/u.data'

def load_movielens(filepath): 
    df = pd.read_csv(filepath, 
                    sep='\t', 
                    names=['user_id', 'item_id', 'rating', 'timestamp'])
    return df 

df = load_movielens(DATA_FILE_PATH)

print(df)
