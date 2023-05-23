from datasets import load_dataset


def common_load_dataset(dataset_name):
    if dataset_name == 'dirty':
        raw_dataset = load_dataset('SetFit/sst2')
        train_dataset = raw_dataset['train']
        val_dataset = raw_dataset['val']
        test_dataset = raw_dataset['test']
    elif dataset_name == 'imdb':
        raw_dataset = load_dataset(dataset_name)
        train_dataset = raw_dataset['train']
        test_dataset = raw_dataset['test']
    else:
        raise NotImplementedError
    return train_dataset, val_dataset, test_dataset