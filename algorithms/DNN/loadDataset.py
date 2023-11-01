import torch


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data, self.labels = data, labels

    def readData(self, data, labels):
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
