import torch


class KnowMANDataSet(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None, adv_labels=None, max_length=200):
        self.encodings = encodings
        self.labels = labels
        self.adv_labels = adv_labels
        self.max_length = max_length

    def __getitem__(self, idx):
        if type(self.adv_labels) != torch.Tensor:
            return {k: torch.tensor(v[idx][0:self.max_length]) for k, v in self.encodings.items()}, \
                       self.labels[idx]
        else:
            return {k: torch.tensor(v[idx][0:self.max_length]) for k, v in self.encodings.items()}, \
                       self.labels[idx], \
                       self.adv_labels[idx]

    def __len__(self):
        return len(self.labels)

