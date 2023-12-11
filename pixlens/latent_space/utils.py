from typing import Optional

import torch
import pickle

def store_latents(image_id: str, latent_type: str, latent_tensor: torch.Tensor , database: dict):
    if image_id not in database:
        database[image_id] = {"positive1": None, "positive2": None, "negative": None, "target": None}
    
    if latent_type in ["positive", "negative", "target"]:
        if latent_type == "positive":
            if database[image_id]["positive1"] is None:
                database[image_id]["positive1"] = latent_tensor
            else:
                database[image_id]["positive2"] = latent_tensor
        else:
            database[image_id][latent_type] = latent_tensor
    
    else:
        raise ValueError("Invalid latent type. Must be one of 'positive', 'negative', 'target'.")

def save_data(database, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(database, file)

def load_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
def transform_database_to_tensors(database: dict, flatten = True):
    X_positive1 = []
    X_positive2 = []
    X_negative = []
    Y = []

    for _, value in database.items():
        if value.values().any(None):
            continue
        if flatten:
            value['positive1'] = value['positive1'].flatten()
            value['positive2'] = value['positive2'].flatten()
            value['negative'] = value['negative'].flatten()
            value['target'] = value['target'].flatten()
        X_positive1.append(value['positive1'])
        X_positive2.append(value['positive2'])
        X_negative.append(value['negative'])
        Y.append(value['target'])

    X_positive1 = torch.stack(X_positive1) if X_positive1 else torch.tensor([])
    X_positive2 = torch.stack(X_positive2) if X_positive2 else torch.tensor([])
    X_negative = torch.stack(X_negative) if X_negative else torch.tensor([])
    Y = torch.stack(Y) if Y else torch.tensor([])

    return X_positive1, X_positive2, X_negative, Y
