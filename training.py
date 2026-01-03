from models.two_tower import TwoTowerModel, BPRLoss
import torch.optim as optim


def train_model(model, train_data, config):

    num_epochs = config['training']['num_epochs']
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    bpr_loss = BPRLoss()
    for epoch in range(num_epochs): 
        for user_ids, pos_items, neg_items, user_feat, pos_item_feat, neg_item_feat in train_data: 
            pos_scores  = model(user_ids, pos_items, user_feat, pos_item_feat)
            neg_scores = model(user_ids, neg_items, user_feat, neg_item_feat)
            
            loss = bpr_loss(pos_scores, neg_scores)
            

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        print(f"Epoch {epoch + 1}, loss: {loss.item():.4f}")
