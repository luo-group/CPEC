import torch
from torch.utils.data import Dataset
import json

class ESMDataset(Dataset):
    def __init__(self, data_file, label_file) -> None:
        self.raw_data = torch.load(data_file)
        self.pids = list(self.raw_data.keys())
        with open(label_file, 'r') as f:
            self.all_labels = json.load(f)
        self.label2idx = {label: i for i, label in enumerate(self.all_labels)}
        self.num_labels = len(self.all_labels)
        self.embeddings = []
        self.labels = []
        for i, data in enumerate(self.raw_data.values()):
            self.embeddings.append(data['embedding'])
            labels = torch.zeros(self.num_labels)
            for label in data['label']:
                labels[self.label2idx[label]] = 1
            self.labels.append(labels)
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, index):
        return self.embeddings[index], self.labels[index]
    
    def get_all_labels(self):
        return self.all_labels
    
    def get_pids(self):
        return self.pidss

if __name__ == '__main__':
    dataset = ESMDataset('../data/mlp_data/valid_data.pt', label_file='../data/mlp_data/all_labels.json')
    print(len(dataset))
    print(dataset[0][0].shape, dataset[0][1].shape)
    print(dataset[0][0])
    print(dataset[0][1])
    
    label1 = dataset[0][1]
    label2 = dataset[1][1]
    label1 = torch.vstack([label1, label1])
    label2 = torch.vstack([label2, label2])
    criterion = torch.nn.BCELoss()
    loss = criterion(label1, label2)
    print(loss)
    # from sklearn.metrics import f1_score
    # print(f1_score(label1, label2, average='micro'))