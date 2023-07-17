from datasets import load_dataset
from DirtDataset import DirtDataset
from DireDataset import DireDataset


def common_load_dataset(dataset_name):
    if dataset_name == 'dirty':
        raw_dataset = load_dataset('Veweew/dirty_small')
        train_dataset = DirtDataset(raw_dataset['train'])
        dev_dataset = DirtDataset(raw_dataset['dev'])
        test_dataset = DirtDataset(raw_dataset['test'])
    elif dataset_name == 'dire':
        raw_dataset = load_dataset('Veweew/dire')
        train_dataset = DireDataset(raw_dataset['train'])
        dev_dataset = DireDataset(raw_dataset['dev'])
        test_dataset = DireDataset(raw_dataset['test'])
    else:
        raise NotImplementedError
    return train_dataset, dev_dataset, test_dataset
