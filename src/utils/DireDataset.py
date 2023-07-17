from src.utils.dire.dataset import Example
from torch.utils.data import Dataset
import json

class DireDataset(Dataset):
    def __init__(self, huggingface_dataset):
        self.huggingface_dataset = huggingface_dataset

    def __getitem__(self, item):
        return Example.from_json_dict(json.loads(self.huggingface_dataset[item]['jsonl']))

    def __len__(self):
        return len(self.huggingface_dataset)

if __name__ == '__main__':
    from utils import common_load_dataset

    train_dataset, dev_dataset, test_dataset = common_load_dataset('dire')

    print(train_dataset[0])
