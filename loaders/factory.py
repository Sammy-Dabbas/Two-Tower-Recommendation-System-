from loaders.movielens_100k import load_movielens_100k 

def get_data_loader(config): 
    loaders = { 
        "movielens-100k": load_movielens_100k ,

           }
    dataset_name = config['dataset']['name']
    loader_fn = loaders[dataset_name]

    return loader_fn(config['dataset']) 
   
 