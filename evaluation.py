import torch



def recall_at_k(model, test_df, train_df, num_items, user_features_lookup, item_feature_lookup, k=10):
    """
    Compute Recall@K: Of all items a user actually likes, what percentage appear in top-K recommendations
    """
    model.eval()
    recalls = []

    #Get the users who have liked items in test set where rating >= 4
    test_liked = test_df[test_df['rating'] >= 4].groupby('user_id')['item_id'].apply(set).to_dict()

    #Get items each user saw during training
    train_seen = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
   
    with torch.no_grad():
        for user_id, liked_items in test_liked.items():
            #Items this user saw in training 
            seen_items = train_seen.get(user_id, set())

            #All items we could recommend 
            all_items = set(range(1, num_items + 1)) - seen_items
            if len(all_items) == 0 or len(liked_items) == 0:
                continue

        
            #creating item and user tensors
            all_items_list = list(all_items)
            num_unseen = len(all_items_list)
            user_features = user_features_lookup[user_id - 1]
            user_tensor = torch.tensor([user_id - 1] * num_unseen, dtype=torch.long)
            user_features_tensor = torch.tensor([user_features] * num_unseen, dtype=torch.float32)
            item_tensor = torch.tensor([item - 1 for item in all_items_list], dtype=torch.long)
          
           
            item_features_list = []
            for item in all_items_list: 
                features_for_this_item = item_feature_lookup[item - 1]
              
                item_features_list.append(features_for_this_item)
            item_features_tensor = torch.tensor(item_features_list, dtype=torch.float32)
            #get scores 
           
            scores  = model(user_tensor, item_tensor, user_features_tensor, item_features_tensor)
        
            k_actual = min(k, len(scores))
            top_scores, top_indices = torch.topk(scores, k_actual)
            top_items = [all_items_list[i] for i in top_indices.tolist()]
          
            #Caculate number of matched between top items and liked items
            matches = set(top_items) & liked_items

            recall = len(matches) / len(liked_items)
            recalls.append(recall)
    #Return average recall across all users
    return sum(recalls) / len(recalls) if recalls else 0.0


def evaluate_bpr(model, test_data):
      model.eval()

      pos_scores_list = []
      neg_scores_list = []

      with torch.no_grad():
         for user_ids, pos_items, neg_items, user_feat, pos_item_feat, neg_item_feat in test_data:              
              pos_scores  = model(user_ids, pos_items, user_feat, pos_item_feat)
              neg_scores = model(user_ids, neg_items, user_feat, neg_item_feat)

              pos_scores_list.append(pos_scores.mean().item())
              neg_scores_list.append(neg_scores.mean().item())

      avg_pos = sum(pos_scores_list) / len(pos_scores_list)
      avg_neg = sum(neg_scores_list) / len(neg_scores_list)

      print(f"Avg positive score: {avg_pos:.4f}")
      print(f"Avg negative score: {avg_neg:.4f}")
      print(f"Difference: {avg_pos - avg_neg:.4f}")


