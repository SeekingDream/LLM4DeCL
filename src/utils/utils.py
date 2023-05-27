from datasets import load_dataset
from src.utils.Example import Example


def common_load_dataset(dataset_name):
    if dataset_name == 'dirty':
        raw_dataset = load_dataset('Veweew/dirty_small')
        train_dataset = raw_dataset['train']
        dev_dataset = raw_dataset['dev']
        test_dataset = raw_dataset['test']
    else:
        raise NotImplementedError
    return train_dataset, dev_dataset, test_dataset
