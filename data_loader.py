import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    
    

def load_all_data(epilepsy_dir, normal_dir, max_seq_length=None):

    epilepsy_files = [os.path.join(epilepsy_dir, f) for f in os.listdir(epilepsy_dir) if f.endswith('.txt')]
    epilepsy_data = []
    
    for file_path in epilepsy_files:
        try:

            data = np.loadtxt(file_path)
            # If a maximum length is specified, clipping or padding is performed
            if max_seq_length is not None:
                if len(data) > max_seq_length:
                    data = data[:max_seq_length]  # Clipping
                elif len(data) < max_seq_length:
                    padding = np.zeros((max_seq_length - len(data)))
                    data = np.concatenate([data, padding])  # Padding
            epilepsy_data.append(data)
        except Exception as e:
            print(f"Unable to load file {file_path}: {e}")
    
    normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith('.txt')]
    normal_data = []
    
    for file_path in normal_files:
        try:
            data = np.loadtxt(file_path)
            if max_seq_length is not None:
                if len(data) > max_seq_length:
                    data = data[:max_seq_length]
                elif len(data) < max_seq_length:
                    padding = np.zeros((max_seq_length - len(data)))
                    data = np.concatenate([data, padding])
            normal_data.append(data)
        except Exception as e:
            print(f"Unable to load file {file_path}: {e}")
    
    #  Create labels (1 for epilepsy, 0 for normal)
    epilepsy_labels = np.ones(len(epilepsy_data))
    normal_labels = np.zeros(len(normal_data))
    

    max_len = max([len(d) for d in epilepsy_data + normal_data])
    padded_data = []
    for d in epilepsy_data + normal_data:
        if len(d) < max_len:
            padded_d = np.pad(d, ((0, max_len - len(d))), 'constant')
            padded_data.append(padded_d)
        else:
            padded_data.append(d)
        
    
    all_data = np.array(padded_data)
    all_labels = np.concatenate([epilepsy_labels, normal_labels])
    
    
    all_data = torch.FloatTensor(all_data).unsqueeze(1)  # [N, 1, seq_len]
    all_labels = torch.LongTensor(all_labels)
    
    return all_data, all_labels

# def prepare_dataloaders(epilepsy_dir, normal_dir, batch_size=32, test_size=0.2, val_size=0.1, max_seq_length=None, random_state=42):

#     all_data, all_labels = load_all_data(epilepsy_dir, normal_dir, max_seq_length)

#     X_train, X_test, y_train, y_test = train_test_split(
#         all_data, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels
#     )

#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train, y_train, test_size=val_size/(1-test_size), random_state=random_state, stratify=y_train
#     )

#     train_dataset = EEGDataset(X_train, y_train)
#     val_dataset = EEGDataset(X_val, y_val)
#     test_dataset = EEGDataset(X_test, y_test)
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
#     return train_loader, val_loader, test_loader







def prepare_dataloaders(epilepsy_dir, normal_dir, batch_size=32, test_size=0.2, val_size=0.1, max_seq_length=None, random_state=42):


    all_data, all_labels = load_all_data(epilepsy_dir, normal_dir, max_seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        all_data, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size/(1-test_size), random_state=random_state, stratify=y_train
    )
    

    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    
    X_full_test = torch.cat([X_train, X_test], dim=0)
    y_full_test = torch.cat([y_train, y_test], dim=0)
    test_dataset = EEGDataset(X_full_test, y_full_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader